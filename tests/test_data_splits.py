"""Tests for window stride and fold resolution."""

from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf

from stock_forecaster.data.dataset import FnspidWindowDataset
from stock_forecaster.data.splits import (
    _resolve_years_field,
    folds_enabled,
    iter_fold_indices,
    materialize_fold_data_cfg,
    resolve_year_splits,
)


def _mini_frame() -> pd.DataFrame:
    rows = []
    for day in range(10):
        rows.append(
            {
                "ticker": "AAA",
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
                "change_pct": 0.1,
                "news": f"news-{day}",
                "target_label": 1 if day % 2 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_window_stride_reduces_sample_count() -> None:
    frame = _mini_frame()
    dense = FnspidWindowDataset(
        frame, seq_len=3, feature_columns=["open", "close"], window_stride=1
    )
    sparse = FnspidWindowDataset(
        frame, seq_len=3, feature_columns=["open", "close"], window_stride=5
    )
    assert len(sparse) < len(dense)
    assert len(dense) == 7
    assert len(sparse) == 2


def test_resolve_year_splits_with_fold() -> None:
    cfg = OmegaConf.create(
        {
            "train_years": [2018],
            "val_years": [2019],
            "test_years": [2023],
            "folds": {
                "enabled": True,
                "fold_index": 1,
                "definitions": [
                    {"name": "a", "train_years": [2018], "val_years": [2019]},
                    {"name": "b", "train_years": [2018, 2019], "val_years": [2020]},
                ],
            },
        }
    )
    train, val, test, name = resolve_year_splits(cfg)
    assert train == [2018, 2019]
    assert val == [2020]
    assert test == [2023]
    assert name == "b"
    assert folds_enabled(cfg)
    assert iter_fold_indices(cfg) == [1]
    assert iter_fold_indices(OmegaConf.merge(cfg, {"folds": {"run_all": True}})) == [0, 1]


def test_year_range_overrides_explicit_list_when_both_set() -> None:
    """Period ablation merges fnspid train_years with period train_year_min/max."""
    cfg = OmegaConf.create(
        {
            "train_years": [2018, 2019, 2020, 2021],
            "train_year_min": 1999,
            "train_year_max": 2001,
        }
    )
    assert _resolve_years_field(cfg, "train_years", "train_year_min", "train_year_max") == [
        1999,
        2000,
        2001,
    ]


def test_resolve_years_from_min_max() -> None:
    cfg = OmegaConf.create({"train_year_min": 2000, "train_year_max": 2003})
    assert _resolve_years_field(cfg, "train_years", "train_year_min", "train_year_max") == [
        2000,
        2001,
        2002,
        2003,
    ]


def test_resolve_year_splits_period_style() -> None:
    cfg = OmegaConf.create(
        {
            "train_year_min": 1999,
            "train_year_max": 2015,
            "val_year_min": 2016,
            "val_year_max": 2017,
            "test_years": [2023],
            "folds": {"enabled": False, "definitions": []},
        }
    )
    train, val, test, fold = resolve_year_splits(cfg)
    assert train[0] == 1999 and train[-1] == 2015
    assert val == [2016, 2017]
    assert test == [2023]
    assert fold is None


def test_materialize_fold_data_cfg() -> None:
    cfg = OmegaConf.create(
        {
            "train_years": [2018],
            "val_years": [2019],
            "test_years": [2023],
            "folds": {
                "enabled": True,
                "fold_index": 0,
                "run_all": True,
                "definitions": [
                    {"name": "a", "train_years": [2018], "val_years": [2019]},
                ],
            },
        }
    )
    folded = materialize_fold_data_cfg(cfg, 0)
    assert folded.folds.fold_index == 0
