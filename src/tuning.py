"""Compact, training-only hyperparameter searches."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def require_boosters():
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost and LightGBM are required for Models 3 and 4. Install dependencies with "
            "'python -m pip install -r requirements.txt' and rerun the project."
        ) from exc
    return XGBClassifier, LGBMClassifier


def tune_models(X_train, y_train, pipeline_steps, cv):
    """Tune each required base learner using training folds only."""
    XGBClassifier, LGBMClassifier = require_boosters()
    models_and_params = {
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [100, 160, 220],
                "model__max_depth": [None, 5, 10, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "SVM": (
            # Calibration supplies predict_proba for soft voting/stacking without
            # using SVC's deprecated probability option.
            CalibratedClassifierCV(SVC(random_state=42), cv=3, method="sigmoid"),
            {
                "model__estimator__C": [0.1, 0.5, 1, 2, 5, 10],
                "model__estimator__kernel": ["rbf", "linear"],
                "model__estimator__gamma": ["scale", "auto", 0.01, 0.1],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42, eval_metric="logloss", n_jobs=1, verbosity=0
            ),
            {
                "model__n_estimators": [80, 140, 200],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1),
            {
                "model__n_estimators": [80, 140, 200],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__num_leaves": [7, 15, 31],
                "model__max_depth": [-1, 3, 5, 8],
                "model__min_child_samples": [5, 10, 20, 30],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
    }
    tuned = {}
    parameters = {}
    for name, (model, distributions) in models_and_params.items():
        pipeline = Pipeline([*pipeline_steps, ("model", model)])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=distributions,
            n_iter=3,
            scoring="accuracy",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        parameters[name] = search.best_params_
    return tuned, parameters
