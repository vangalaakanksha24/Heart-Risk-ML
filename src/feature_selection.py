"""Training-only feature selection pipelines."""
from __future__ import annotations

import numpy as np
from functools import partial
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def feature_count(total_features: int) -> int:
    """Select a compact, data-dependent subset without hard-coding feature names."""
    return max(2, min(total_features, max(5, min(20, int(np.ceil(total_features * 0.60))))))


def build_selector(method: str, n_features: int):
    if method == "Mutual Information":
        return SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=n_features)
    if method == "RFE":
        estimator = LogisticRegression(max_iter=3000, solver="liblinear", random_state=42)
        return RFE(estimator=estimator, n_features_to_select=n_features, step=0.2)
    if method == "L1 Model-Based":
        estimator = LinearSVC(
            penalty="l1", dual=False, C=1.0, max_iter=3000, random_state=42
        )
        return SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=n_features)
    raise ValueError(f"Unknown feature-selection method: {method}")


def selected_feature_names(preprocessor, general_engineer, selector) -> list[str]:
    """Read names after fitted transformations and a fitted selector."""
    preprocessed_names = preprocessor.get_feature_names_out()
    engineered_names = general_engineer.get_feature_names_out(preprocessed_names)
    mask = selector.get_support()
    return list(np.asarray(engineered_names, dtype=object)[mask])
