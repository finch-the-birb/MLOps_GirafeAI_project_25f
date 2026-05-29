"""Validation-set probability threshold search for binary direction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from stock_forecaster.lightning_module import StockForecaster


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f_beta: float


def _binary_metrics(
    labels: np.ndarray, preds: np.ndarray, *, beta: float = 1.0
) -> ThresholdMetrics:
    labels = labels.astype(int)
    preds = preds.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    beta2 = float(beta) * float(beta)
    denom = beta2 * precision + recall
    # F_beta = (1+beta^2)*P*R / (beta^2*P + R)
    f_beta = (1.0 + beta2) * precision * recall / max(denom, 1e-12)
    return ThresholdMetrics(
        threshold=0.0,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        f_beta=float(f_beta),
    )


def _news_batch_input(batch: dict[str, object]) -> list[str] | list[list[str]]:
    """Early fusion: collated ``news_text`` is list[list[str]]; late fusion: list[str]."""
    news_text = batch["news_text"]
    if news_text and isinstance(news_text[0], list):
        return news_text  # type: ignore[return-value]
    return news_text  # type: ignore[return-value]


def collect_probabilities(
    forecaster: StockForecaster,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    forecaster.eval()
    forecaster.to(device)
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            time_series = batch["time_series"].to(device)
            news_input = _news_batch_input(batch)
            labels = batch["label"].to(device)
            logits = forecaster(time_series, news_input)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    low: float = 0.35,
    high: float = 0.65,
    step: float = 0.02,
    objective: str = "f1",
    beta: float = 1.0,
) -> tuple[ThresholdMetrics, list[ThresholdMetrics]]:
    grid = np.arange(low, high + step * 0.5, step)
    results: list[ThresholdMetrics] = []
    for threshold in grid:
        preds = (probs >= threshold).astype(int)
        metrics = _binary_metrics(labels, preds, beta=beta)
        results.append(
            ThresholdMetrics(
                threshold=float(threshold),
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                f_beta=metrics.f_beta,
            )
        )
    key = {
        "f1": "f1",
        "f_beta": "f_beta",
        "precision": "precision",
        "accuracy": "accuracy",
    }[objective]
    best = max(results, key=lambda item: getattr(item, key))
    return best, results
