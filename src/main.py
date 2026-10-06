"""Terminal entry point for the leakage-safe heart disease ML project."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from data_understanding import (
    describe_dataset,
    find_target_column,
    load_dataset,
    normalize_target,
    print_data_understanding,
)
from evaluation import (
    calculate_metrics,
    cv_metrics,
    diagnostic_metrics,
    print_best_model_confusion_matrix,
    print_metric_block,
    print_metrics_table,
    save_best_model_diagnostic_charts,
    save_feature_selection_chart,
)
from feature_engineering import (
    add_clinical_features,
    build_general_engineer,
    print_clinical_report,
)
from feature_selection import (
    build_selector,
    feature_count,
    selected_feature_names,
)
from models import build_model_collection
from prediction import offer_patient_prediction
from preprocessing import (
    build_preprocessor,
    find_invalid_values,
    iqr_outlier_summary,
    print_preprocessing_summary,
)
from tuning import tune_models


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

RANDOM_STATE = 42


def score_cv(estimator, X, y, cv):
    result = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
    )
    return cv_metrics(result)


def feature_engineering_comparison(X_train, y_train, cv):
    """Compare general feature engineering on training folds only."""

    results = {}

    for method in ("Polynomial Features", "PowerTransformer"):

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(X_train),
                ),
                (
                    "engineering",
                    build_general_engineer(method),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        results[method] = score_cv(
            pipeline,
            X_train,
            y_train,
            cv,
        )

    best = max(
        results,
        key=lambda name: (
            results[name]["Accuracy"],
            results[name]["F1-Score"],
        ),
    )

    return results, best


def feature_selection_comparison(
    X_train,
    y_train,
    engineering_method,
    cv,
):
    """Compare selection methods solely with training-fold transformations."""

    # Fit only on the training partition to set
    # an adaptive selection size.
    probe_preprocessor = build_preprocessor(X_train)

    probe_features = probe_preprocessor.fit_transform(
        X_train,
        y_train,
    )

    probe_engineer = build_general_engineer(
        engineering_method
    )

    transformed = probe_engineer.fit_transform(
        probe_features,
        y_train,
    )

    count = feature_count(
        transformed.shape[1]
    )

    results = {}

    for method in (
        "Mutual Information",
        "RFE",
        "L1 Model-Based",
    ):

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(X_train),
                ),
                (
                    "engineering",
                    build_general_engineer(
                        engineering_method
                    ),
                ),
                (
                    "selector",
                    build_selector(
                        method,
                        count,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        results[method] = score_cv(
            pipeline,
            X_train,
            y_train,
            cv,
        )

    best = max(
        results,
        key=lambda name: (
            results[name]["Accuracy"],
            results[name]["F1-Score"],
        ),
    )

    # This fit is only for reporting feature names.
    # It does not use test data.
    fitted_preprocessor = build_preprocessor(
        X_train
    )

    training_matrix = fitted_preprocessor.fit_transform(
        X_train,
        y_train,
    )

    fitted_engineer = build_general_engineer(
        engineering_method
    )

    engineered_matrix = fitted_engineer.fit_transform(
        training_matrix,
        y_train,
    )

    fitted_selector = build_selector(
        best,
        count,
    )

    fitted_selector.fit(
        engineered_matrix,
        y_train,
    )

    names = selected_feature_names(
        fitted_preprocessor,
        fitted_engineer,
        fitted_selector,
    )

    return (
        results,
        best,
        count,
        names,
    )


def make_pipeline_steps(
    X_train,
    engineering_method,
    selector_method,
    n_selected,
):
    """Return fresh, unfitted pipeline steps for every model."""

    return [
        (
            "preprocessor",
            build_preprocessor(X_train),
        ),
        (
            "engineering",
            build_general_engineer(
                engineering_method
            ),
        ),
        (
            "selector",
            build_selector(
                selector_method,
                n_selected,
            ),
        ),
    ]


def main() -> int:

    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "heart_2020_cleaned.csv"
    )

    print("=" * 60)
    print("HEART DISEASE PREDICTION ML PROJECT")
    print("=" * 60)

    try:

        # ==================================================
        # DATA LOADING
        # ==================================================

        data = load_dataset(
            dataset_path
        )

        # ==================================================
        # DATA UNDERSTANDING
        # ==================================================

        target_column = find_target_column(
            data
        )

        info = describe_dataset(
            data,
            target_column,
        )

        print_data_understanding(
            info,
            target_column,
        )

        # ==================================================
        # DUPLICATE REMOVAL
        # ==================================================

        data = data.drop_duplicates().copy()

        removed_duplicates = info[
            "duplicates"
        ]

        # ==================================================
        # TARGET NORMALIZATION
        # ==================================================

        y = normalize_target(
            data.pop(target_column)
        )

        X = data

        # ==================================================
        # INVALID VALUES
        # ==================================================

        invalid_values = find_invalid_values(
            X
        )

        # ==================================================
        # OUTLIER ANALYSIS
        # ==================================================

        outliers = iqr_outlier_summary(
            X,
            info["numerical"],
        )

        print_preprocessing_summary(
            removed_duplicates,
            invalid_values,
            outliers,
        )

        # ==================================================
        # TRAIN / TEST SPLIT
        # ==================================================

        X_train_raw, X_test_raw, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=RANDOM_STATE,
            )
        )

        # ==================================================
        # CLINICAL FEATURE ENGINEERING
        # ==================================================

        X_train, clinical_report = (
            add_clinical_features(
                X_train_raw
            )
        )

        X_test, _ = add_clinical_features(
            X_test_raw
        )

        print(
            "\nTRAIN-TEST SPLIT\n"
            "----------------"
        )

        print(
            f"Training Samples: {len(X_train)}"
        )

        print(
            f"Testing Samples: {len(X_test)}"
        )

        print(
            "Split: 80% training / 20% testing"
        )

        print(
            "Stratification: target labels; "
            "random_state=42"
        )

        # ==================================================
        # CROSS VALIDATION
        # ==================================================

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        # ==================================================
        # FEATURE ENGINEERING COMPARISON
        # ==================================================

        (
            fe_results,
            best_engineering,
        ) = feature_engineering_comparison(
            X_train,
            y_train,
            cv,
        )

        print_metrics_table(
            "\nFEATURE ENGINEERING COMPARISON",
            fe_results,
        )

        print(
            f"Best Feature Engineering Method: "
            f"{best_engineering}"
        )

        print_clinical_report(
            clinical_report
        )

        # ==================================================
        # FEATURE SELECTION COMPARISON
        # ==================================================

        (
            fs_results,
            best_selector,
            n_selected,
            names,
        ) = feature_selection_comparison(
            X_train,
            y_train,
            best_engineering,
            cv,
        )

        print_metrics_table(
            "\nFEATURE SELECTION COMPARISON",
            fs_results,
            index_name="Feature Selection Strategy",
        )

        print(
            f"Best Feature Selection Method: "
            f"{best_selector}"
        )

        print("Selected Features:")

        for index, name in enumerate(
            names,
            start=1,
        ):
            print(
                f"{index}. {name}"
            )

        # ==================================================
        # HYPERPARAMETER OPTIMIZATION
        # ==================================================

        print(
            "\nHYPERPARAMETER OPTIMIZATION\n"
            "---------------------------"
        )

        pipeline_steps = make_pipeline_steps(
            X_train,
            best_engineering,
            best_selector,
            n_selected,
        )

        tuned, parameters = tune_models(
            X_train,
            y_train,
            pipeline_steps,
            cv,
        )

        for name, params in parameters.items():

            print(
                f"{name} Best Parameters: "
                f"{params}"
            )

        # ==================================================
        # MODEL COLLECTION
        # ==================================================

        candidates = build_model_collection(
            tuned["Random Forest"],
            tuned["SVM"],
            tuned["XGBoost"],
            tuned["LightGBM"],
            cv,
        )

        # ==================================================
        # MODEL RESULTS
        # ==================================================

        print(
            "\nMODEL RESULTS\n"
            "============="
        )

        test_results = {}
        test_diagnostics = {}

        for name, model in candidates.items():

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            test_results[name] = calculate_metrics(
                y_test,
                predictions,
            )

            test_diagnostics[name] = diagnostic_metrics(
                y_test,
                predictions,
                probabilities,
            )

            print_metric_block(
                name,
                test_results[name],
            )

        # ==================================================
        # FINAL MODEL COMPARISON
        # ==================================================

        print_metrics_table(
            "\nFINAL MODEL COMPARISON\n"
            "======================",
            test_results,
        )

        # ==================================================
        # BEST MODEL
        # ==================================================

        best_model = max(
            test_results,
            key=lambda name: (
                test_results[name]["Accuracy"],
                test_results[name]["F1-Score"],
                test_results[name]["Precision"],
                test_results[name]["Recall"],
            ),
        )

        best = test_results[
            best_model
        ]

        print_best_model_confusion_matrix(
            best_model,
            test_diagnostics[best_model],
        )

        feature_selection_path = save_feature_selection_chart(
            fs_results,
            project_root / "outputs",
        )

        confusion_path, roc_path = save_best_model_diagnostic_charts(
            best_model,
            test_diagnostics[best_model],
            project_root / "outputs",
        )

        print("\nDIAGNOSTIC CHARTS\n=================")
        print(f"Feature selection scores: {feature_selection_path}")
        print(f"Best-model confusion matrix: {confusion_path}")
        print(f"Best-model ROC curve: {roc_path}")

        print(
            "\nBEST MODEL\n"
            "=========="
        )

        print(
            f"Best Model: {best_model}"
        )

        print(
            f"Best Test Accuracy: "
            f"{best['Accuracy']:.2f}%"
        )

        print(
            f"Best Precision: "
            f"{best['Precision']:.2f}%"
        )

        print(
            f"Best Recall: "
            f"{best['Recall']:.2f}%"
        )

        print(
            f"Best F1-Score: "
            f"{best['F1-Score']:.2f}%"
        )

        offer_patient_prediction(
            candidates[best_model],
            X,
            add_clinical_features,
        )

        return 0

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
