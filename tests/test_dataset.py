"""Unit tests for point-in-time window construction."""

import pandas as pd

from stock_forecaster.data.dataset import FnspidWindowDataset


def test_sliding_window_builds_samples() -> None:
    rows = []
    for day in range(40):
        rows.append(
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0 + day * 0.01,
                "volume": 1000.0,
                "change_pct": 0.1,
                "news": f"headline-{day}",
                "target_label": 1,
            }
        )
    frame = pd.DataFrame(rows)
    dataset = FnspidWindowDataset(
        frame,
        seq_len=5,
        feature_columns=["open", "high", "low", "close", "volume", "change_pct"],
        years=[2020],
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["time_series"].shape == (5, 6)
    assert isinstance(sample["news_text"], str)
    assert len(sample["daily_news"]) == 5


def test_early_fusion_daily_news_per_timestep() -> None:
    rows = []
    for day in range(10):
        rows.append(
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1000.0,
                "change_pct": 0.1,
                "news": f"headline-{day}",
                "target_label": 0,
            }
        )
    frame = pd.DataFrame(rows)
    dataset = FnspidWindowDataset(
        frame,
        seq_len=3,
        feature_columns=["open", "high", "low", "close", "volume", "change_pct"],
        years=[2020],
        fusion_mode="early",
    )
    sample = dataset[0]
    assert len(sample["daily_news"]) == 3
    assert sample["daily_news"][0] == "headline-0"
