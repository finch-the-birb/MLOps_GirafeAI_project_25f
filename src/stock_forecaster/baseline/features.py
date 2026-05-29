"""Flatten sliding-window OHLCV tensors for sklearn / tree baselines."""

from __future__ import annotations

import numpy as np

from stock_forecaster.data.dataset import FnspidWindowDataset


def dataset_to_xy(dataset: FnspidWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a tabular matrix from windowed samples (no text).

    Each row is ``seq_len * n_features`` values from ``time_series``, row-major
    (day 0 features, day 1 features, …).
    """
    if len(dataset) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

    n_features = int(dataset.samples[0].time_series.shape[-1])
    flat_dim = dataset.seq_len * n_features
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for sample in dataset.samples:
        window = sample.time_series.reshape(-1).astype(np.float32)
        if window.shape[0] != flat_dim:
            msg = f"Expected flat dim {flat_dim}, got {window.shape[0]}"
            raise ValueError(msg)
        x_rows.append(window)
        y_rows.append(int(sample.label))
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.int32)
