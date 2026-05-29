"""Tests for tabular baseline feature extraction."""

from __future__ import annotations

import pandas as pd

from stock_forecaster.baseline.features import dataset_to_xy
from stock_forecaster.data.dataset import FnspidWindowDataset


def test_dataset_to_xy_flattens_window() -> None:
    rows = []
    for day in range(35):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "ticker": "AAA",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100.0,
                "change_pct": 0.1,
                "news": "",
                "target_label": day % 2,
            }
        )
    frame = pd.DataFrame(rows)
    ds = FnspidWindowDataset(
        frame,
        seq_len=5,
        feature_columns=["open", "high", "low", "close", "volume", "change_pct"],
        years=[2020],
        window_stride=5,
        fusion_mode="early",
    )
    x, y = dataset_to_xy(ds)
    assert x.shape[1] == 5 * 6
    assert y.ndim == 1
    assert len(y) == len(ds)
