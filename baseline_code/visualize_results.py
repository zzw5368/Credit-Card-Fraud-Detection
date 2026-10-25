"""Visualize locally saved DS340W fraud-detection ensemble results.

Usage (from this script's directory):
    python visualize_results.py

Requires results/model_comparison.csv and per-model predictions saved by
1st_code.py. Displays figures and saves PNGs to results/charts/.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

MODELS = ["AdaBoost_RF_KNN", "XGBoost_RF_KNN"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC_AP"]


def main():
    parser = argparse.ArgumentParser(description="Visualize AdaBoost vs. XGBoost results")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results", help="Directory containing experiment results")
    parser.add_argument("--no-show", action="store_true", help="Save PNG charts without opening plot windows")
    args = parser.parse_args()
    results_dir = args.results.resolve()
    charts_dir = results_dir / "charts"

    comparison_file = results_dir / "model_comparison.csv"
    if not comparison_file.is_file():
        raise FileNotFoundError(f"Missing {comparison_file}. Run 1st_code.py first, using the same results directory.")
    comparison = pd.read_csv(comparison_file, index_col="Model")
    for name in MODELS:
        if name not in comparison.index:
            raise ValueError(f"Missing model in comparison results: {name}")
    missing_metrics = [c for c in METRICS + ["TN", "FP", "FN", "TP"] if c not in comparison]
    if missing_metrics:
        raise ValueError(f"Missing comparison columns: {missing_metrics}")

    predictions = {}
    for name in MODELS:
        path = results_dir / f"{name}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing predictions: {path}")
        predictions[name] = pd.read_csv(path)
        for col in ("actual", "fraud_probability"):
            if col not in predictions[name]:
                raise ValueError(f"Missing '{col}' in {path}")
    charts_dir.mkdir(parents=True, exist_ok=True)

    def save_and_show(fig, filename):
        fig.tight_layout()
        target = charts_dir / filename
        fig.savefig(target, dpi=300, bbox_inches="tight")
        if not args.no_show:
            plt.show()
        plt.close(fig)
        print("Saved:", target)

    print("\n" + "=" * 65)
    print("CREDIT CARD FRAUD DETECTION — ADABOOST VS XGBOOST")
    print("=" * 65)
    print("Baseline: AdaBoost + Random Forest + KNN")
    print("Improved: Bayesian-optimized XGBoost + Random Forest + KNN")
    print("Sampling: SMOTE + ENN (training-only)")
    print("\nMODEL PERFORMANCE TABLE\n")
    print(comparison.loc[MODELS, METRICS].round(4).to_string())

    fig, ax = plt.subplots(figsize=(13, 6))
    comparison.loc[MODELS, METRICS].T.plot(kind="bar", ax=ax, rot=0)
    ax.set(title="AdaBoost vs XGBoost Ensemble Performance", ylabel="Score", xlabel="Evaluation Metric", ylim=(0, 1.05))
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    save_and_show(fig, "model_performance.png")

    for name in MODELS:
        row = comparison.loc[name]
        matrix = np.array([[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]])
        fig, ax = plt.subplots(figsize=(7, 5))
        plot = ax.imshow(matrix, cmap="Blues")
        ax.set(title=f"Confusion Matrix — {name}", xlabel="Predicted Class", ylabel="Actual Class")
        ax.set_xticks([0, 1], labels=["Normal", "Fraud"])
        ax.set_yticks([0, 1], labels=["Normal", "Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        fig.colorbar(plot, ax=ax)
        save_and_show(fig, f"{name}_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODELS:
        data = predictions[name]
        fpr, tpr, _ = roc_curve(data["actual"], data["fraud_probability"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(title="ROC Curve Comparison", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    save_and_show(fig, "roc_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODELS:
        data = predictions[name]
        precision, recall, _ = precision_recall_curve(data["actual"], data["fraud_probability"])
        ap = comparison.loc[name, "PR_AUC_AP"]
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
    ax.set(title="Precision-Recall Curve Comparison", xlabel="Recall", ylabel="Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    save_and_show(fig, "precision_recall_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.loc[MODELS, ["FP", "FN"]].plot(kind="bar", ax=ax, rot=0)
    ax.set(title="False Positive and False Negative Comparison", ylabel="Number of Transactions", xlabel="Model")
    ax.legend(["False Positives", "False Negatives"])
    ax.grid(axis="y", alpha=0.3)
    save_and_show(fig, "error_comparison.png")

    baseline, improved = (comparison.loc[name] for name in MODELS)
    print("\n" + "=" * 65)
    print("FINAL RESEARCH SUMMARY")
    print("=" * 65)
    for metric in METRICS:
        difference = improved[metric] - baseline[metric]
        print(f"{metric}: AdaBoost={baseline[metric]:.4f}, XGBoost={improved[metric]:.4f}, Difference={difference:+.4f}")
    print(f"False negatives: AdaBoost={int(baseline['FN'])}, XGBoost={int(improved['FN'])}")
    print(f"False positives: AdaBoost={int(baseline['FP'])}, XGBoost={int(improved['FP'])}")
    if improved["PR_AUC_AP"] > baseline["PR_AUC_AP"]:
        print("XGBoost achieved higher average precision on this test set.")
    elif improved["PR_AUC_AP"] < baseline["PR_AUC_AP"]:
        print("AdaBoost achieved higher average precision on this test set.")
    else:
        print("Both ensembles achieved equal average precision on this test set.")
    print("These findings do not establish statistical significance or final-validation performance.")
    print("Charts saved to:", charts_dir)


if __name__ == "__main__":
    main()
