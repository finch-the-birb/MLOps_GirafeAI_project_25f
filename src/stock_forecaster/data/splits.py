"""Temporal year splits and walk-forward fold resolution for Hydra configs."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _as_year_list(years: list[int] | ListConfig | tuple[int, ...]) -> list[int]:
    return [int(y) for y in years]


def _resolve_years_field(
    data_cfg: DictConfig,
    list_key: str,
    min_key: str,
    max_key: str,
) -> list[int]:
    """Use inclusive min/max when set (period presets); else explicit year list (fnspid default)."""
    if min_key in data_cfg and max_key in data_cfg:
        start = int(data_cfg[min_key])
        end = int(data_cfg[max_key])
        if end < start:
            msg = f"data.{max_key} ({end}) must be >= data.{min_key} ({start})"
            raise ValueError(msg)
        return list(range(start, end + 1))
    years_list = data_cfg.get(list_key)
    if years_list is not None and len(years_list) > 0:
        return _as_year_list(years_list)
    msg = (
        f"Set data.{list_key} or both data.{min_key} and data.{max_key} "
        f"(got keys: {list(data_cfg.keys())})"
    )
    raise ValueError(msg)


def get_fold_definitions(data_cfg: DictConfig) -> list[dict[str, Any]]:
    """Return fold dicts with keys: name, train_years, val_years."""
    folds_cfg = data_cfg.get("folds")
    if folds_cfg is None:
        return []
    definitions = folds_cfg.get("definitions")
    if definitions is None or len(definitions) == 0:
        return []
    return [OmegaConf.to_container(fold, resolve=True) for fold in definitions]  # type: ignore[return-value]


def folds_enabled(data_cfg: DictConfig) -> bool:
    folds_cfg = data_cfg.get("folds")
    if folds_cfg is None:
        return False
    return bool(folds_cfg.get("enabled", False)) and len(get_fold_definitions(data_cfg)) > 0


def resolve_year_splits(data_cfg: DictConfig) -> tuple[list[int], list[int], list[int], str | None]:
    """
    Resolve train/val/test years and optional fold name.

    When folds.enabled is true, train_years and val_years come from
    folds.definitions[fold_index]; test_years always from data_cfg.test_years.
    """
    test_years = _as_year_list(data_cfg.test_years)

    if not folds_enabled(data_cfg):
        train_years = _resolve_years_field(
            data_cfg, "train_years", "train_year_min", "train_year_max"
        )
        val_years = _resolve_years_field(data_cfg, "val_years", "val_year_min", "val_year_max")
        return train_years, val_years, test_years, None

    definitions = get_fold_definitions(data_cfg)
    fold_index = int(data_cfg.folds.get("fold_index", 0))
    if fold_index < 0 or fold_index >= len(definitions):
        msg = (
            f"fold_index={fold_index} out of range for {len(definitions)} folds. "
            "Set data.folds.fold_index or disable data.folds.enabled."
        )
        raise IndexError(msg)

    fold = definitions[fold_index]
    train_years = _as_year_list(fold["train_years"])
    val_years = _as_year_list(fold["val_years"])
    fold_name = str(fold.get("name", f"fold_{fold_index}"))
    return train_years, val_years, test_years, fold_name


def materialize_fold_data_cfg(data_cfg: DictConfig, fold_index: int) -> DictConfig:
    """Copy data config with folds.fold_index set (for multi-fold training loops)."""
    cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    if not folds_enabled(data_cfg):
        return cfg
    cfg.folds.fold_index = fold_index
    return cfg


def iter_fold_indices(data_cfg: DictConfig) -> list[int]:
    """Indices to run when data.folds.run_all is true."""
    if not folds_enabled(data_cfg):
        return [0]
    if not bool(data_cfg.folds.get("run_all", False)):
        return [int(data_cfg.folds.get("fold_index", 0))]
    return list(range(len(get_fold_definitions(data_cfg))))
