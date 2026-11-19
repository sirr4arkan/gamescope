"""Clean the original VG Sales dataset, train regressors, and save artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "vgsales.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FEATURE_COLUMNS = ["Platform", "Genre", "Year", "Publisher"]
CATEGORICAL_COLUMNS = ["Platform", "Genre", "Publisher"]
NUMERIC_COLUMNS = ["Year"]
TARGET_COLUMN = "Global_Sales"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def clean_dataset(data: pd.DataFrame):
    """Return a reproducibly cleaned copy and a row-count audit trail."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))

    counts = {"rows_before_cleaning": int(len(data))}
    result = data.copy()
    before = len(result)
    result = result.drop_duplicates()
    counts["identical_duplicates_removed"] = int(before - len(result))
    counts["rows_after_duplicate_removal"] = int(len(result))

    before = len(result)
    result = result.dropna(subset=REQUIRED_COLUMNS).copy()
    counts["missing_required_values_removed"] = int(before - len(result))
    counts["rows_after_missing_value_removal"] = int(len(result))

    for column in ("Year", TARGET_COLUMN):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    before = len(result)
    result = result.dropna(subset=["Year", TARGET_COLUMN]).copy()
    counts["invalid_numeric_values_removed"] = int(before - len(result))
    counts["rows_after_numeric_validation"] = int(len(result))

    finite_values = np.isfinite(result[["Year", TARGET_COLUMN]]).all(axis=1)
    valid_target = result[TARGET_COLUMN] >= 0
    before = len(result)
    result = result.loc[finite_values & valid_target].copy()
    counts["invalid_or_negative_target_rows_removed"] = int(before - len(result))
    counts["rows_after_cleaning"] = int(len(result))
    if result.empty:
        raise ValueError("No rows remain after data cleaning.")
    return result, counts


def build_pipeline(estimator):
    """Keep categorical encoding and estimation together for fit and inference."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessing), ("estimator", estimator)]
    )


def train_and_evaluate(dataset_path=DATASET_PATH, artifacts_dir=ARTIFACTS_DIR):
    data = pd.read_csv(dataset_path)
    cleaned, cleaning_counts = clean_dataset(data)
    structure = {
        "columns": data.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in data.isna().sum().items()
        },
    }
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = artifacts_dir / "cleaned_vgsales.csv"
    cleaned.to_csv(cleaned_path, index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned[FEATURE_COLUMNS],
        cleaned[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
    )
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }
    evaluations = {}
    evaluations_after_clipping = {}
    fitted_models = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        evaluations[name] = {
            "r2": float(r2_score(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
        }
        clipped_predictions = np.maximum(predictions, 0)
        evaluations_after_clipping[name] = {
            "r2": float(r2_score(y_test, clipped_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, clipped_predictions))),
            "mae": float(mean_absolute_error(y_test, clipped_predictions)),
        }
        fitted_models[name] = pipeline

    best_name = min(
        evaluations,
        key=lambda name: (evaluations[name]["rmse"], -evaluations[name]["r2"]),
    )
    model_path = artifacts_dir / "model_vgsales.joblib"
    joblib.dump(fitted_models[best_name], model_path)

    metadata = {
        "dataset": str(Path(dataset_path).resolve().relative_to(ROOT_DIR))
        if Path(dataset_path).resolve().is_relative_to(ROOT_DIR)
        else Path(dataset_path).name,
        "source_structure": structure,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "numeric_features": NUMERIC_COLUMNS,
        "cleaning": cleaning_counts,
        "train_test_split": {"test_size": 0.2, "random_state": 42},
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "evaluations": evaluations,
        "evaluations_after_nonnegative_clipping": evaluations_after_clipping,
        "selected_model": best_name,
        "model_artifact": model_path.name,
    }
    with (artifacts_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    report_lines = [
        "# EDA dan Pembersihan Dataset VG Sales",
        "",
        f"- Dataset sumber: `{metadata['dataset']}`",
        f"- Baris mentah: {cleaning_counts['rows_before_cleaning']:,}",
        f"- Baris bersih: {cleaning_counts['rows_after_cleaning']:,}",
        f"- Kolom: {', '.join(data.columns)}",
        "",
        "## Tipe Data dan Missing Values Sumber",
        "",
        "| Kolom | Tipe | Missing |",
        "|---|---|---:|",
    ]
    report_lines.extend(
        f"| {column} | {structure['dtypes'][column]} | "
        f"{structure['missing_values'][column]} |"
        for column in structure["columns"]
    )
    report_lines.extend(
        [
            "",
            "## Audit Pembersihan",
            "",
            "| Tahap | Jumlah baris |",
            "|---|---:|",
            f"| Sebelum pembersihan | {cleaning_counts['rows_before_cleaning']:,} |",
            f"| Duplikat identik dihapus | "
            f"{cleaning_counts['identical_duplicates_removed']:,} |",
            f"| Missing pada kolom wajib dihapus | "
            f"{cleaning_counts['missing_required_values_removed']:,} |",
            f"| Nilai numerik tidak valid dihapus | "
            f"{cleaning_counts['invalid_numeric_values_removed']:,} |",
            f"| Target negatif/tidak finite dihapus | "
            f"{cleaning_counts['invalid_or_negative_target_rows_removed']:,} |",
            f"| Sesudah pembersihan | {cleaning_counts['rows_after_cleaning']:,} |",
            "",
            "Duplikat hanya dihapus jika seluruh kolom identik. Nama game tidak "
            "digunakan untuk menghapus baris. Nilai ekstrem tidak dihapus.",
        ]
    )
    (ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "reports" / "eda_summary.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result, indent=2))
