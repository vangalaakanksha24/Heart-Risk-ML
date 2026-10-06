"""General and deterministic clinical feature engineering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer


@dataclass(frozen=True)
class ClinicalFeatureReport:
    created: list[str]
    skipped: list[str]


def add_clinical_features(features: pd.DataFrame) -> tuple[pd.DataFrame, ClinicalFeatureReport]:
    """Add only deterministic interactions supported by actual source columns."""
    result = features.copy()
    lower_to_original = {column.lower().replace(" ", "_"): column for column in features.columns}
    created: list[str] = []
    skipped: list[str] = []
    age = lower_to_original.get("age")
    heart_rate = lower_to_original.get("max_heart_rate")
    cholesterol = lower_to_original.get("cholesterol")

    if age:
        result["age_group"] = pd.cut(
            result[age], bins=[-np.inf, 44, 54, 64, np.inf], labels=["<45", "45-54", "55-64", "65+"]
        ).astype(object)
        created.append("age_group (age categories)")
    else:
        skipped.append("age_group (no age/AgeCategory column)")
    if age and heart_rate:
        safe_age = result[age].replace(0, np.nan)
        result["hr_ratio"] = result[heart_rate] / safe_age
        created.append("hr_ratio (max heart rate divided by age)")
    else:
        skipped.append("hr_ratio (requires age and max heart rate)")
    if age and cholesterol:
        result["chol_age"] = result[cholesterol] * result[age]
        created.append("chol_age (cholesterol * age)")
    else:
        skipped.append("chol_age (requires cholesterol and age)")
    skipped.append("high_risk (skipped: no validated dataset-specific risk definition)")
    return result, ClinicalFeatureReport(created, skipped)


class PolynomialTransformer(BaseEstimator, TransformerMixin):
    """Polynomial expansion with sklearn-compatible feature-name propagation."""

    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X, y=None):
        self.transformer_ = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformer_.get_feature_names_out(input_features)


def build_general_engineer(method: str) -> Pipeline:
    """Build an unfitted post-preprocessing general feature transformer."""
    if method == "Polynomial Features":
        return Pipeline([("polynomial", PolynomialTransformer(degree=2))])
    if method == "PowerTransformer":
        return Pipeline([("power", PowerTransformer(method="yeo-johnson", standardize=True))])
    raise ValueError(f"Unknown feature-engineering method: {method}")


def print_clinical_report(report: ClinicalFeatureReport) -> None:
    print("\nCLINICAL FEATURE ENGINEERING\n----------------------------")
    print(f"Created Features: {report.created or 'None'}")
    print(f"Skipped Features: {report.skipped or 'None'}")
