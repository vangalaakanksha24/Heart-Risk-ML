"""Model and ensemble construction."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def base_estimators(random_forest, svm, xgboost, lightgbm):
    return {
        "rf": random_forest,
        "svm": svm,
        "xgb": xgboost,
        "lgbm": lightgbm,
    }


def build_model_collection(random_forest, svm, xgboost, lightgbm, cv):
    """Return the four required candidate architectures."""
    return {
        "Model 1: Random Forest": random_forest,
        "Model 2: RF + SVM": VotingClassifier(
            estimators=[("rf", random_forest), ("svm", svm)], voting="soft"
        ),
        "Model 3: RF + SVM + XGBoost": StackingClassifier(
            estimators=[("rf", random_forest), ("svm", svm), ("xgb", xgboost)],
            final_estimator=LogisticRegression(max_iter=3000, random_state=42),
            cv=cv,
            stack_method="predict_proba",
        ),
        "Model 4: RF + SVM + XGBoost + LightGBM": StackingClassifier(
            estimators=[("rf", random_forest), ("svm", svm), ("xgb", xgboost), ("lgbm", lightgbm)],
            final_estimator=LogisticRegression(max_iter=3000, random_state=42),
            cv=cv,
            stack_method="predict_proba",
        ),
    }
