"""Dataset inspection and target validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TARGET_CANDIDATES = ("HeartDisease", "target", "Target", "heart_disease", "Heart_Disease")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a supplied CSV without changing its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - message is for terminal users
        raise RuntimeError(f"Could not read dataset '{path}': {exc}") from exc


def find_target_column(data: pd.DataFrame) -> str:
    """Find a verified target column, preferring HeartDisease when supplied."""
    if "HeartDisease" in data.columns:
        return "HeartDisease"
    matches = [name for name in TARGET_CANDIDATES if name in data.columns]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "No recognized target column was found. Expected HeartDisease or a "
            f"known target name; available columns: {list(data.columns)}"
        )
    raise ValueError(f"Ambiguous target columns found: {matches}")


def normalize_target(target: pd.Series) -> pd.Series:
    """Convert only an observed No/Yes target to 0/1; otherwise retain valid binary labels."""
    values = set(target.dropna().unique())
    if values == {"No", "Yes"}:
        return target.map({"No": 0, "Yes": 1}).astype(int)
    if values.issubset({0, 1, False, True}):
        return target.astype(int)
    raise ValueError(
        "Target must be binary. Observed non-missing values: "
        f"{sorted(map(str, values))}"
    )


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Treat low-cardinality numeric codes as categorical, otherwise preserve numeric columns."""
    categorical: list[str] = []
    numerical: list[str] = []
    for column in features.columns:
        series = features[column]
        if not pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 10:
            categorical.append(column)
        else:
            numerical.append(column)
    return numerical, categorical


def describe_dataset(data: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Return programmatic understanding information for terminal presentation."""
    features = data.drop(columns=[target_column])
    numerical, categorical = infer_feature_types(features)
    return {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.astype(str),
        "missing": data.isna().sum(),
        "duplicates": int(data.duplicated().sum()),
        "numerical": numerical,
        "categorical": categorical,
        "categorical_values": {
            column: sorted(map(str, features[column].dropna().unique()))
            for column in categorical
        },
        "target_distribution": data[target_column].value_counts(dropna=False).sort_index(),
        "statistics": data.describe(include="all").transpose(),
    }


def print_data_understanding(info: dict[str, Any], target_column: str) -> None:
    print("\nDATA UNDERSTANDING\n------------------")
    print(f"Dataset Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print("Data Types:")
    print(info["dtypes"].to_string())
    print(f"Numerical Features: {info['numerical']}")
    print(f"Categorical Features: {info['categorical']}")
    print("Missing Values:")
    print(info["missing"].to_string())
    print(f"Duplicate Rows: {info['duplicates']}")
    print("Unique Values for Categorical Columns:")
    for column, values in info["categorical_values"].items():
        print(f"  {column}: {values}")
    print(f"Target Column: {target_column}")
    print("Target Class Distribution:")
    print(info["target_distribution"].to_string())
    print("Basic Statistical Information:")
    print(info["statistics"].to_string())
