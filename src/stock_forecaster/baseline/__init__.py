"""Tabular baselines (OHLCV-only) for comparison with multimodal models."""

from stock_forecaster.baseline.features import dataset_to_xy
from stock_forecaster.baseline.metrics import classification_metrics

__all__ = ["classification_metrics", "dataset_to_xy"]
