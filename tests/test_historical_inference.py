"""Tests for dynamic inference window construction."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from stock_forecaster.data.inference_window import (
    build_dynamic_inference_sample,
    validate_target_date_2023,
)
from stock_forecaster.service.historical import (
    build_historical_inference_payload,
    find_test_sample,
)


def _data_cfg(path: Path, *, seq_len: int = 5) -> OmegaConf:
    return OmegaConf.create(
        {
            "processed_file": str(path),
            "seq_len": seq_len,
            "feature_columns": ["open", "high", "low", "close", "volume", "change_pct"],
            "max_news_per_window": 32,
            "max_news_chars": 512,
            "max_news_chars_per_day": 256,
            "label_threshold_pct": 0.3,
        }
    )


def _row(ticker: str, day: pd.Timestamp, close: float, news: str = "n") -> dict:
    return {
        "ticker": ticker,
        "date": day,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
        "change_pct": 0.1,
        "news": news,
        "target_label": 1,
    }


def test_validate_target_date_2023() -> None:
    assert validate_target_date_2023("2023-06-01") == date(2023, 6, 1)
    with pytest.raises(ValueError, match="2023"):
        validate_target_date_2023("2022-06-01")


def test_dynamic_window_uses_2022_tail(tmp_path: Path) -> None:
    rows = []
    for i in range(10):
        rows.append(_row("AAPL", pd.Timestamp("2022-12-15") + pd.Timedelta(days=i), 100 + i))
    for i in range(20):
        rows.append(_row("AAPL", pd.Timestamp("2023-01-03") + pd.Timedelta(days=i), 200 + i))
    path = tmp_path / "sample.parquet"
    pd.DataFrame(rows).to_parquet(path)
    cfg = _data_cfg(path, seq_len=5)

    sample = build_dynamic_inference_sample("AAPL", "2023-01-03", cfg)
    assert sample.window_start_date.startswith("2022-")
    assert sample.record.time_series.shape == (5, 6)


def test_find_test_sample_any_trading_day(tmp_path: Path) -> None:
    rows = []
    for day in range(40):
        rows.append(
            _row("AAPL", pd.Timestamp("2023-01-03") + pd.Timedelta(days=day), 1.0 + day * 0.01)
        )
    path = tmp_path / "sample.parquet"
    pd.DataFrame(rows).to_parquet(path)
    cfg = _data_cfg(path, seq_len=5)

    record, news_count, outcome = find_test_sample("aapl", "2023-01-10", cfg)
    assert record.ticker == "AAPL"
    assert record.target_date == "2023-01-10"
    assert news_count >= 1
    assert outcome.forward_return_pct is not None

    with pytest.raises(ValueError, match="No trading row"):
        find_test_sample("AAPL", "2023-01-01", cfg)


def test_build_historical_inference_payload_early(tmp_path: Path) -> None:
    rows = []
    for day in range(40):
        rows.append(_row("AAPL", pd.Timestamp("2023-01-03") + pd.Timedelta(days=day), 1.0))
    path = tmp_path / "sample.parquet"
    pd.DataFrame(rows).to_parquet(path)
    cfg = _data_cfg(path, seq_len=5)
    fake = np.zeros((1, 5, 32), dtype=np.float32)
    with patch("stock_forecaster.service.historical.encode_early_per_step", return_value=fake):
        payload = build_historical_inference_payload(
            "AAPL", "2023-01-10", cfg, object(), fusion_mode="early"
        )
    assert payload.text_per_step is not None
    assert payload.outcome is not None
