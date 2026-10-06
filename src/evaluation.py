"""Metric calculation and terminal tables."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-Score"]


def calculate_metrics(y_true, predictions) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, predictions) * 100,
        "Precision": precision_score(y_true, predictions, zero_division=0) * 100,
        "Recall": recall_score(y_true, predictions, zero_division=0) * 100,
        "F1-Score": f1_score(y_true, predictions, zero_division=0) * 100,
    }


def diagnostic_metrics(y_true, predictions, probabilities) -> dict[str, object]:
    """Calculate test-only diagnostic values after a model has been finalized."""
    return {
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_true, probabilities) * 100,
        "roc_curve": roc_curve(y_true, probabilities),
    }


def cv_metrics(cv_result: dict) -> dict[str, float]:
    return {
        "Accuracy": cv_result["test_accuracy"].mean() * 100,
        "Precision": cv_result["test_precision"].mean() * 100,
        "Recall": cv_result["test_recall"].mean() * 100,
        "F1-Score": cv_result["test_f1"].mean() * 100,
    }


def print_metric_block(model_name: str, metrics: dict[str, float]) -> None:
    print(f"\n{model_name}")
    for metric in METRIC_COLUMNS:
        print(f"{metric}: {metrics[metric]:.2f}%")


def print_metrics_table(title: str, rows: dict[str, dict[str, float]], index_name: str = "Model") -> None:
    print(f"\n{title}")
    table = pd.DataFrame.from_dict(rows, orient="index")[METRIC_COLUMNS]
    formatted = table.map(lambda value: f"{value:.2f}%")
    formatted.index.name = index_name
    print(formatted.to_string())


def print_best_model_confusion_matrix(model_name: str, diagnostic: dict[str, object]) -> None:
    """Print the labelled test-set confusion matrix for the selected best model."""
    matrix = diagnostic["confusion_matrix"]
    print("\nBEST MODEL CONFUSION MATRIX\n===========================")
    print(model_name)
    print("                 Predicted No  Predicted Yes")
    print(f"Actual No        {matrix[0, 0]:<12} {matrix[0, 1]}")
    print(f"Actual Yes       {matrix[1, 0]:<12} {matrix[1, 1]}")


def save_feature_selection_chart(rows: dict[str, dict[str, float]], output_dir):
    """Save a horizontal Precision/Recall/F1 comparison by feature-selection strategy."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(rows)
    metrics = ("Precision", "Recall", "F1-Score")
    positions = np.arange(len(strategies))
    offsets = (-0.22, 0.0, 0.22)
    palette = {
        "Precision": "#2563EB",
        "Recall": "#EA580C",
        "F1-Score": "#16A34A",
    }
    all_scores = [rows[strategy][metric] for strategy in strategies for metric in metrics]
    score_min = max(0, min(all_scores) - 0.45)
    score_max = min(100, max(all_scores) + 0.65)

    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for metric, offset in zip(metrics, offsets):
        scores = [rows[strategy][metric] for strategy in strategies]
        metric_positions = positions + offset
        axis.hlines(
            metric_positions,
            score_min,
            scores,
            color=palette[metric],
            alpha=0.28,
            linewidth=3,
        )
        axis.scatter(
            scores,
            metric_positions,
            s=125,
            color=palette[metric],
            edgecolors="white",
            linewidths=1.5,
            label=metric,
            zorder=3,
        )
        for score, position in zip(scores, metric_positions):
            axis.annotate(
                f"{score:.2f}%",
                (score, position),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                fontweight="semibold",
                color=palette[metric],
            )
    axis.set_yticks(positions, strategies)
    axis.set_xlabel("Score (%)")
    axis.set_ylabel("Feature Selection Strategies")
    axis.set_xlim(score_min, score_max)
    axis.set_ylim(-0.45, len(strategies) - 1 + 0.65)
    axis.set_title("Precision, Recall, and F1-Score by Feature Selection Strategy", pad=14)
    axis.legend(loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    axis.grid(axis="x", alpha=0.3, linestyle="--")
    axis.set_axisbelow(True)
    axis.text(
        0.5,
        -0.16,
        "Zoomed score range makes the small performance differences easy to compare.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    chart_path = output_dir / "feature_selection_scores.png"
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return chart_path


def save_best_model_diagnostic_charts(model_name: str, diagnostic: dict[str, object], output_dir) -> tuple[object, object]:
    """Save a confusion matrix and ROC curve for only the selected best model."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = diagnostic["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Confusion Matrix — {model_name}")
    axis.set_xticks([0, 1], ["No", "Yes"])
    axis.set_yticks([0, 1], ["No", "Yes"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    confusion_path = output_dir / "confusion_matrix_best_model.png"
    figure.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    false_positive_rate, true_positive_rate, _ = diagnostic["roc_curve"]
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"{model_name} (AUC = {diagnostic['roc_auc']:.2f}%)",
    )
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    axis.set_title(f"ROC Curve — {model_name}")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    roc_path = output_dir / "roc_curve_best_model.png"
    figure.savefig(roc_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return confusion_path, roc_path
