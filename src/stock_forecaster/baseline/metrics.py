"""Classification metrics for tabular baselines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true_i = y_true.astype(np.int32)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_i, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def best_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    for thr in grid:
        m = classification_metrics(y_true, y_prob, threshold=float(thr))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics
