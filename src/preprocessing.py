"""Leakage-safe preprocessing utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_understanding import infer_feature_types


def find_invalid_values(features: pd.DataFrame) -> dict[str, int]:
    """Report values that are invalid by dataset representation, without clinical guesswork."""
    invalid: dict[str, int] = {}
    for column in features.select_dtypes(include=np.number).columns:
        name = column.lower().replace(" ", "_")
        # These measurements cannot be negative. Zero is retained because it may be a coded value.
        if any(token in name for token in ("age", "bp", "cholesterol", "heart_rate")):
            count = int((features[column] < 0).sum())
            if count:
                invalid[column] = count
    return invalid


def iqr_outlier_summary(features: pd.DataFrame, numerical_columns: list[str]) -> dict[str, int]:
    """Describe IQR outliers only; valid medical observations are never removed or clipped."""
    summary: dict[str, int] = {}
    for column in numerical_columns:
        series = pd.to_numeric(features[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            summary[column] = 0
        else:
            summary[column] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    return summary


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted transformer. All learned stages fit inside model pipelines."""
    numerical, categorical = infer_feature_types(features)
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numerical), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def print_preprocessing_summary(
    removed_duplicates: int,
    invalid_values: dict[str, int],
    outliers: dict[str, int],
) -> None:
    print("\nDATA PREPROCESSING\n------------------")
    print("Missing-value handling: median imputation (numeric) and most-frequent imputation (categorical), fitted only in pipelines.")
    print(f"Duplicate handling: {removed_duplicates} duplicate row(s) removed before splitting.")
    print(f"Invalid-value handling: {invalid_values or 'No representation-invalid negative values detected.'}")
    print("Outlier analysis (IQR; reported only, no rows removed or clipped):")
    print(outliers or "  No numerical features available for analysis.")
    print("Encoding: one-hot encoding with unknown categories ignored, fitted only in pipelines.")
    print("Scaling: StandardScaler for numerical features, fitted only in pipelines.")
