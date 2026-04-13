"""Evaluation metrics: macro F1, AUROC, precision, recall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


@dataclass
class EvalMetrics:
    """Computed evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_macro: float
    auroc: float | None  # None if only one class present
    confusion: list[list[int]]

    def summary(self) -> str:
        lines = [
            f"Accuracy:  {self.accuracy:.3f}",
            f"Precision: {self.precision:.3f}",
            f"Recall:    {self.recall:.3f}",
            f"Macro F1:  {self.f1_macro:.3f}",
        ]
        if self.auroc is not None:
            lines.append(f"AUROC:     {self.auroc:.3f}")
        lines.append(f"Confusion: {self.confusion}")
        return "\n".join(lines)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float] | None = None,
) -> EvalMetrics:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_scores: Optional confidence scores for AUROC.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    auroc = None
    if y_scores is not None and len(set(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return EvalMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_macro=f1,
        auroc=auroc,
        confusion=cm,
    )
