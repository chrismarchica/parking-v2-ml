"""Model evaluation utilities for parking ticket prediction."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_xgb import ParkingTicketModel
from features import FeaturePipeline


class ModelEvaluator:
    """Comprehensive model evaluation for parking ticket prediction."""

    def __init__(self, model: ParkingTicketModel):
        """
        Initialize evaluator.

        Args:
            model: Trained ParkingTicketModel instance.
        """
        self.model = model
        self.target_encoder = model.label_encoders.get(model.target)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        top_k: list[int] = [1, 3, 5],
    ) -> dict:
        """
        Run comprehensive evaluation.

        Args:
            X: Feature matrix.
            y_true: True labels (encoded).
            top_k: List of k values for top-k accuracy.

        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
        }

        # Top-k accuracy
        for k in top_k:
            if k <= y_proba.shape[1]:
                results[f"top_{k}_accuracy"] = top_k_accuracy_score(
                    y_true, y_proba, k=k
                )

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        results["per_class"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        # Macro/weighted averages
        for avg in ["macro", "weighted"]:
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average=avg
            )
            results[f"{avg}_precision"] = p
            results[f"{avg}_recall"] = r
            results[f"{avg}_f1"] = f

        return results

    def plot_confusion_matrix(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        top_n: int = 20,
        figsize: tuple = (14, 12),
        save_path: str = None,
    ) -> None:
        """
        Plot confusion matrix for top-N classes.

        Args:
            X: Feature matrix.
            y_true: True labels.
            top_n: Number of top classes to show.
            figsize: Figure size.
            save_path: Path to save figure.
        """
        y_pred = self.model.predict(X)

        # Get top N classes by frequency
        class_counts = pd.Series(y_true).value_counts()
        top_classes = class_counts.head(top_n).index.tolist()

        # Filter to top classes
        mask = np.isin(y_true, top_classes)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # Compute confusion matrix
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)

        # Get class labels
        if self.target_encoder:
            class_labels = [
                self.target_encoder.inverse_transform([c])[0]
                for c in sorted(set(y_true_filtered))
            ]
        else:
            class_labels = sorted(set(y_true_filtered))

        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Top {top_n} Classes)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: tuple = (10, 8),
        save_path: str = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to show.
            figsize: Figure size.
            save_path: Path to save figure.
        """
        importance = self.model.get_feature_importance().head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(importance["feature"], importance["importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def generate_report(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        output_path: str = None,
    ) -> str:
        """
        Generate a text evaluation report.

        Args:
            X: Feature matrix.
            y_true: True labels.
            output_path: Path to save report.

        Returns:
            Report string.
        """
        results = self.evaluate(X, y_true)
        y_pred = self.model.predict(X)

        lines = [
            "=" * 60,
            "MODEL EVALUATION REPORT",
            "=" * 60,
            "",
            f"Overall Accuracy: {results['accuracy']:.4f}",
            f"Top-3 Accuracy:   {results.get('top_3_accuracy', 'N/A'):.4f}",
            f"Top-5 Accuracy:   {results.get('top_5_accuracy', 'N/A'):.4f}",
            "",
            f"Macro F1:     {results['macro_f1']:.4f}",
            f"Weighted F1:  {results['weighted_f1']:.4f}",
            "",
            "-" * 60,
            "Classification Report",
            "-" * 60,
            "",
        ]

        # Get class labels
        if self.target_encoder:
            target_names = self.target_encoder.classes_
        else:
            target_names = None

        report = classification_report(y_true, y_pred, target_names=target_names)
        lines.append(report)

        lines.extend([
            "",
            "-" * 60,
            "Top 15 Features",
            "-" * 60,
            "",
        ])

        for _, row in self.model.get_feature_importance().head(15).iterrows():
            lines.append(f"  {row['feature']:30s} {row['importance']:.4f}")

        report_str = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_str)

        return report_str


def run_evaluation(model_path: str, data_sample: float = 0.1):
    """
    Run evaluation on a saved model.

    Args:
        model_path: Path to saved model directory.
        data_sample: Sample fraction of data to use.
    """
    print("Loading model...")
    model = ParkingTicketModel.load(model_path)

    print("Loading evaluation data...")
    pipeline = FeaturePipeline()
    df = pipeline.load_and_transform(sample_frac=data_sample)
    X, y = pipeline.prepare_for_training(df, target=model.target)

    X_enc = model.prepare_features(X)
    y_enc = model.encode_target(y)

    print("Running evaluation...")
    evaluator = ModelEvaluator(model)
    report = evaluator.generate_report(X_enc, y_enc)
    print(report)

    return evaluator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate parking ticket model")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("--sample", type=float, default=0.1, help="Data sample fraction")

    args = parser.parse_args()
    run_evaluation(args.model_path, args.sample)

