import json
import logging
import math
import pickle
from functools import lru_cache

import joblib
import pandas as pd
from django.conf import settings
from django.db import DatabaseError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ml.train_model import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from predictor.ai_explainer import (
    ANALYSIS_INSTRUCTIONS,
    AIExplanationError,
    generate_analysis_explanation,
)
from predictor.models import PredictionLog


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(settings.MODEL_PATH)


@lru_cache(maxsize=1)
def _load_metadata():
    data = pd.read_csv(settings.CLEANED_DATA_PATH, usecols=FEATURE_COLUMNS + ["Global_Sales"])
    model_metadata = json.loads(
        settings.MODEL_METADATA_PATH.read_text(encoding="utf-8")
    )
    year_range = [
        int(data["Year"].min()),
        int(data["Year"].max()),
    ]

    def aggregate_sales(column):
        grouped = (
            data.groupby(column)["Global_Sales"]
            .agg(["mean", "size"])
            .rename(columns={"mean": "average", "size": "observations"})
            .sort_values("average", ascending=False)
        )
        categories = [
            {
                "category": str(category),
                "average_global_sales": round(float(row["average"]), 4),
                "observations": int(row["observations"]),
            }
            for category, row in grouped.iterrows()
        ]
        return {
            "averages": {
                item["category"]: item["average_global_sales"]
                for item in categories
            },
            "counts": {
                item["category"]: item["observations"]
                for item in categories
            },
            "categories": categories,
        }

    genre_sales = aggregate_sales("Genre")
    platform_sales = aggregate_sales("Platform")
    cleaned_rows = int(len(data))
    return {
        "platforms": sorted(data["Platform"].dropna().astype(str).unique().tolist()),
        "genres": sorted(data["Genre"].dropna().astype(str).unique().tolist()),
        "publishers": sorted(data["Publisher"].dropna().astype(str).unique().tolist()),
        "year_range": year_range,
        "average_sales_by_genre": genre_sales["averages"],
        "average_sales_by_platform": platform_sales["averages"],
        "sales_analysis": {
            "genre_analysis": {
                "category_name": "Genre",
                "categories": genre_sales["categories"],
                "total_observations": cleaned_rows,
                "year_range": year_range,
            },
            "platform_analysis": {
                "category_name": "Platform",
                "categories": platform_sales["categories"],
                "total_observations": cleaned_rows,
                "year_range": year_range,
            },
        },
        "sales_observations_by_genre": genre_sales["counts"],
        "sales_observations_by_platform": platform_sales["counts"],
        "dataset_summary": {
            "cleaned_rows": cleaned_rows,
            "mean_global_sales": float(data["Global_Sales"].mean()),
            "median_global_sales": float(data["Global_Sales"].median()),
            "max_global_sales": float(data["Global_Sales"].max()),
        },
        "model_summary": {
            "selected_model": model_metadata["selected_model"],
            "training_rows": model_metadata["training_rows"],
            "test_rows": model_metadata["test_rows"],
            "train_test_split": model_metadata["train_test_split"],
            "raw_metrics": model_metadata["evaluations"][
                model_metadata["selected_model"]
            ],
            "clipped_metrics": model_metadata[
                "evaluations_after_nonnegative_clipping"
            ][model_metadata["selected_model"]],
        },
    }


def _json_error(message, status=400):
    return JsonResponse({"error": message}, status=status)


def _success_category(sales):
    if sales < 1:
        return "Niche"
    if sales < 5:
        return "Standard"
    return "Blockbuster"


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return _json_error("Use POST for this endpoint.", 405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _json_error("Request body must be valid JSON.")
    if not isinstance(payload, dict):
        return _json_error("Request body must be a JSON object.")

    title = payload.get("game_title", "")
    platform = payload.get("platform")
    genre = payload.get("genre")
    publisher = payload.get("publisher")
    year = payload.get("release_year")
    if title is None:
        title = ""
    if not isinstance(title, str) or len(title) > 200:
        return _json_error("game_title must be text no longer than 200 characters.")
    if any(not isinstance(value, str) or not value.strip() for value in (platform, genre, publisher)):
        return _json_error("platform, genre, and publisher are required non-empty strings.")
    if isinstance(year, bool) or not isinstance(year, int):
        return _json_error("release_year must be an integer.")

    try:
        allowed = _load_metadata()
    except (OSError, ValueError, KeyError):
        logger.exception("Could not load cleaned dataset metadata.")
        return _json_error("Pilihan data belum tersedia. Coba lagi nanti.", 503)
    if platform not in allowed["platforms"]:
        return _json_error("platform is not available in the cleaned dataset.")
    if genre not in allowed["genres"]:
        return _json_error("genre is not available in the cleaned dataset.")
    if publisher not in allowed["publishers"]:
        return _json_error("publisher is not available in the cleaned dataset.")
    minimum_year, maximum_year = allowed["year_range"]
    if not minimum_year <= year <= maximum_year:
        return _json_error(f"release_year must be between {minimum_year} and {maximum_year}.")

    try:
        model = _load_model()
        row = pd.DataFrame(
            [{"Platform": platform, "Genre": genre, "Year": year, "Publisher": publisher}],
            columns=FEATURE_COLUMNS,
        )
        prediction_value = max(0.0, float(model.predict(row)[0]))
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        ValueError,
        TypeError,
        AttributeError,
    ):
        logger.exception("Prediction model could not process the request.")
        return _json_error("Prediksi belum dapat diproses. Coba lagi nanti.", 503)
    if not math.isfinite(prediction_value):
        return _json_error("Model returned a non-finite prediction.", 503)

    category = _success_category(prediction_value)
    try:
        log = PredictionLog.objects.create(
            game_title=title.strip(),
            platform=platform,
            genre=genre,
            publisher=publisher,
            release_year=year,
            predicted_global_sales=prediction_value,
        )
    except DatabaseError:
        logger.exception("Could not save prediction history.")
        return _json_error("Prediksi tidak dapat disimpan. Coba lagi nanti.", 503)
    return JsonResponse(
        {
            "id": log.id,
            "game_title": log.game_title,
            "platform": log.platform,
            "genre": log.genre,
            "publisher": log.publisher,
            "release_year": log.release_year,
            "predicted_global_sales": prediction_value,
            "category": category,
        }
    )


@csrf_exempt
def explain(request):
    if request.method != "POST":
        return _json_error("Use POST for this endpoint.", 405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _json_error("Request body must be valid JSON.")
    if not isinstance(payload, dict) or set(payload) != {"diagram_type"}:
        return _json_error("Request hanya boleh berisi diagram_type.")

    analysis_type = payload.get("diagram_type")
    if not isinstance(analysis_type, str) or analysis_type not in ANALYSIS_INSTRUCTIONS:
        return _json_error("Jenis analisis tidak dikenali.")

    try:
        analysis_context = _load_metadata()["sales_analysis"][analysis_type]
    except (OSError, ValueError, KeyError):
        logger.exception("Could not load market analysis data.")
        return _json_error("Data analisis belum dapat dimuat. Coba lagi nanti.", 503)

    try:
        explanation = generate_analysis_explanation(analysis_type, analysis_context)
    except AIExplanationError as error:
        logger.warning("Market analysis explanation failed (%s).", error.error_code)
        return _json_error(error.public_message, error.status_code)

    return JsonResponse(
        {
            "diagram_type": analysis_type,
            "explanation": explanation,
        }
    )


def history(request):
    if request.method != "GET":
        return _json_error("Use GET for this endpoint.", 405)
    try:
        rows = PredictionLog.objects.all()[:20]
        history_items = [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(),
                "game_title": row.game_title,
                "platform": row.platform,
                "genre": row.genre,
                "publisher": row.publisher,
                "release_year": row.release_year,
                "predicted_global_sales": row.predicted_global_sales,
            }
            for row in rows
        ]
    except DatabaseError:
        logger.exception("Could not load prediction history.")
        return _json_error("Riwayat belum dapat dimuat. Coba lagi nanti.", 503)
    return JsonResponse({"history": history_items})


def metadata(request):
    if request.method != "GET":
        return _json_error("Use GET for this endpoint.", 405)
    try:
        return JsonResponse(_load_metadata())
    except (OSError, ValueError, KeyError):
        logger.exception("Could not load cleaned dataset metadata.")
        return _json_error("Pilihan data belum tersedia. Coba lagi nanti.", 503)
