import json
import math
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ui.errors import ApiError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLEANED_DATA_PATH = ARTIFACTS_DIR / "cleaned_vgsales.csv"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
MODEL_PATH = ARTIFACTS_DIR / "model_vgsales.joblib"
FEATURE_COLUMNS = ["Platform", "Genre", "Year", "Publisher"]
TARGET_COLUMN = "Global_Sales"


def _aggregate_sales(data, column):
    grouped = (
        data.groupby(column)[TARGET_COLUMN]
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
            item["category"]: item["average_global_sales"] for item in categories
        },
        "counts": {
            item["category"]: item["observations"] for item in categories
        },
        "categories": categories,
    }


@st.cache_data(ttl=3600)
def _load_metadata():
    if not CLEANED_DATA_PATH.is_file() or not MODEL_METADATA_PATH.is_file():
        raise ApiError(
            "Artifact dataset atau metadata tidak tersedia. Pastikan folder "
            "artifacts/ ikut disertakan saat deploy."
        )
    try:
        data = pd.read_csv(
            CLEANED_DATA_PATH,
            usecols=FEATURE_COLUMNS + [TARGET_COLUMN],
        )
        model_metadata = json.loads(
            MODEL_METADATA_PATH.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ApiError(
            "Dataset bersih atau metadata model tidak dapat dibaca."
        ) from error

    if not isinstance(model_metadata, dict):
        raise ApiError("Format metadata model tidak valid.")
    try:
        year_range = [
            int(data["Year"].min()),
            int(data["Year"].max()),
        ]
        selected_model = model_metadata["selected_model"]
        raw_metrics = model_metadata["evaluations"][selected_model]
        clipped_metrics = model_metadata[
            "evaluations_after_nonnegative_clipping"
        ][selected_model]
        training_rows = int(model_metadata["training_rows"])
        test_rows = int(model_metadata["test_rows"])
        train_test_split = float(model_metadata["train_test_split"]["test_size"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ApiError("Struktur metadata model tidak lengkap atau tidak valid.") from error

    genre_sales = _aggregate_sales(data, "Genre")
    platform_sales = _aggregate_sales(data, "Platform")
    cleaned_rows = int(len(data))
    return {
        "platforms": sorted(data["Platform"].dropna().astype(str).unique().tolist()),
        "genres": sorted(data["Genre"].dropna().astype(str).unique().tolist()),
        "publishers": sorted(
            data["Publisher"].dropna().astype(str).unique().tolist()
        ),
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
            "mean_global_sales": float(data[TARGET_COLUMN].mean()),
            "median_global_sales": float(data[TARGET_COLUMN].median()),
            "max_global_sales": float(data[TARGET_COLUMN].max()),
        },
        "model_summary": {
            "selected_model": selected_model,
            "training_rows": training_rows,
            "test_rows": test_rows,
            "train_test_split": train_test_split,
            "raw_metrics": raw_metrics,
            "clipped_metrics": clipped_metrics,
        },
    }


@st.cache_resource
def _load_model():
    if not MODEL_PATH.is_file():
        raise ApiError(
            "Artifact model tidak tersedia. Pastikan folder artifacts/ ikut "
            "disertakan saat deploy."
        )
    try:
        return joblib.load(MODEL_PATH)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        ValueError,
        TypeError,
        AttributeError,
    ) as error:
        raise ApiError("Artifact model tidak dapat dimuat oleh runtime ini.") from error


def get_metadata():
    return _load_metadata()


def predict_locally(payload):
    if not isinstance(payload, dict):
        raise ApiError("Data prediksi harus berupa objek.")

    title = payload.get("game_title", "")
    platform = payload.get("platform")
    genre = payload.get("genre")
    publisher = payload.get("publisher")
    year = payload.get("release_year")

    if title is None:
        title = ""
    if not isinstance(title, str) or len(title) > 200:
        raise ApiError("Nama game harus berupa teks maksimal 200 karakter.")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (platform, genre, publisher)
    ):
        raise ApiError("Platform, genre, dan publisher wajib diisi.")
    if isinstance(year, bool) or not isinstance(year, int):
        raise ApiError("Tahun rilis harus berupa bilangan bulat.")

    metadata = get_metadata()
    if platform not in metadata["platforms"]:
        raise ApiError("Platform tidak tersedia pada dataset bersih.")
    if genre not in metadata["genres"]:
        raise ApiError("Genre tidak tersedia pada dataset bersih.")
    if publisher not in metadata["publishers"]:
        raise ApiError("Publisher tidak tersedia pada dataset bersih.")
    minimum_year, maximum_year = metadata["year_range"]
    if not minimum_year <= year <= maximum_year:
        raise ApiError(
            f"Tahun rilis harus berada antara {minimum_year} dan {maximum_year}."
        )

    features = pd.DataFrame(
        [
            {
                "Platform": platform,
                "Genre": genre,
                "Year": year,
                "Publisher": publisher,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    try:
        prediction_value = max(0.0, float(_load_model().predict(features)[0]))
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
    ) as error:
        raise ApiError("Model belum dapat memproses skenario ini.") from error
    if not math.isfinite(prediction_value):
        raise ApiError("Model menghasilkan nilai prediksi yang tidak valid.")

    category = (
        "Niche"
        if prediction_value < 1
        else "Standard"
        if prediction_value < 5
        else "Blockbuster"
    )
    return {
        "game_title": title.strip(),
        "platform": platform,
        "genre": genre,
        "publisher": publisher,
        "release_year": year,
        "predicted_global_sales": prediction_value,
        "category": category,
    }
