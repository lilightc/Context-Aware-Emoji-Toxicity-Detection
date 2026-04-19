"""Evaluation metrics with bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricWithCI:
    """A metric value with an optional bootstrap confidence interval."""
    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    def format(self, width: int = 7) -> str:
        if self.ci_low is None:
            return f"{self.value:.3f}"
        return f"{self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"


@dataclass
class EvalMetrics:
    """Computed evaluation metrics with CIs."""
    accuracy: MetricWithCI
    precision: MetricWithCI
    recall: MetricWithCI
    f1_macro: MetricWithCI
    auroc: MetricWithCI | None
    confusion: list[list[int]]
    n_samples: int
    n_bootstrap: int = 0

    def summary(self) -> str:
        label = "CI" if self.n_bootstrap > 0 else "point"
        lines = [
            f"n samples: {self.n_samples}  (bootstrap={self.n_bootstrap})",
            f"Accuracy:  {self.accuracy.format()}",
            f"Precision: {self.precision.format()}",
            f"Recall:    {self.recall.format()}",
            f"Macro F1:  {self.f1_macro.format()}",
        ]
        if self.auroc is not None:
            lines.append(f"AUROC:     {self.auroc.format()}")
        lines.append(f"Confusion: {self.confusion}  (values shown with 95% {label})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def pack(m: MetricWithCI | None) -> dict | None:
            if m is None:
                return None
            return {"value": m.value, "ci_low": m.ci_low, "ci_high": m.ci_high}

        return {
            "n_samples": self.n_samples,
            "n_bootstrap": self.n_bootstrap,
            "accuracy": pack(self.accuracy),
            "precision": pack(self.precision),
            "recall": pack(self.recall),
            "f1_macro": pack(self.f1_macro),
            "auroc": pack(self.auroc),
            "confusion_matrix": self.confusion,
        }


def _point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None,
) -> dict[str, float | None]:
    """Compute point estimates for all metrics from numpy arrays."""
    out = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": None,
    }
    if y_scores is not None and len(set(y_true.tolist())) > 1:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            pass
    return out


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float] | None = None,
    n_bootstrap: int = 1000,
    rng_seed: int = 0,
) -> EvalMetrics:
    """Compute classification metrics with percentile bootstrap 95% CIs.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_scores: Optional confidence scores for AUROC.
        n_bootstrap: Number of bootstrap resamples. 0 skips CI computation.
        rng_seed: Seed for the bootstrap RNG (for reproducible CIs).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_scores_arr = np.asarray(y_scores) if y_scores is not None else None
    n = len(y_true_arr)

    point = _point_metrics(y_true_arr, y_pred_arr, y_scores_arr)

    boot_dist: dict[str, list[float]] = {"accuracy": [], "precision": [], "recall": [], "f1_macro": [], "auroc": []}
    if n_bootstrap > 0 and n > 1:
        rng = np.random.default_rng(rng_seed)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yt, yp = y_true_arr[idx], y_pred_arr[idx]
            ys = y_scores_arr[idx] if y_scores_arr is not None else None
            bm = _point_metrics(yt, yp, ys)
            for k in boot_dist:
                if bm[k] is not None:
                    boot_dist[k].append(bm[k])

    def wrap(name: str) -> MetricWithCI | None:
        if point[name] is None:
            return None
        if not boot_dist[name]:
            return MetricWithCI(value=point[name])
        lo, hi = np.percentile(boot_dist[name], [2.5, 97.5])
        return MetricWithCI(value=point[name], ci_low=float(lo), ci_high=float(hi))

    return EvalMetrics(
        accuracy=wrap("accuracy"),
        precision=wrap("precision"),
        recall=wrap("recall"),
        f1_macro=wrap("f1_macro"),
        auroc=wrap("auroc"),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        n_samples=n,
        n_bootstrap=n_bootstrap if n > 1 else 0,
    )
