"""Fusion-mode resolution for inference services."""

from __future__ import annotations

from typing import Literal

from omegaconf import DictConfig

FusionMode = Literal["late", "early"]


def resolve_fusion_mode(
    requested: str | None,
    data_cfg: DictConfig,
    inference_cfg: DictConfig | None = None,
) -> FusionMode:
    """Pick fusion mode from request override, inference config, then data config."""
    for candidate in (
        requested,
        None if inference_cfg is None else inference_cfg.get("fusion_mode"),
        data_cfg.get("fusion_mode"),
        "late",
    ):
        if candidate is None:
            continue
        mode = str(candidate).strip().lower()
        if mode in ("late", "early"):
            return mode  # type: ignore[return-value]
    msg = "fusion_mode must be 'late' or 'early'"
    raise ValueError(msg)


def validate_model_fusion_compatibility(fusion_mode: FusionMode, model_cfg: DictConfig) -> None:
    """Ensure the loaded Hydra model matches the requested fusion path."""
    target = str(model_cfg.get("_target_", ""))
    is_early_model = "EarlyFusionStockModel" in target
    if fusion_mode == "early" and not is_early_model:
        msg = (
            "Early fusion was requested but the API was started with a late-fusion model. "
            "Restart with e.g. model=multimodal_early data.fusion_mode=early, or pass "
            "fusion_mode=late."
        )
        raise ValueError(msg)
    if fusion_mode == "late" and is_early_model:
        msg = (
            "Late fusion was requested but the API was started with an early-fusion model. "
            "Restart with model=multimodal, or pass fusion_mode=early."
        )
        raise ValueError(msg)


def triton_model_name(inference_cfg: DictConfig, fusion_mode: FusionMode) -> str:
    """Return the Triton model repository name for the given fusion mode."""
    if fusion_mode == "early":
        return str(inference_cfg.get("triton_model_name_early", inference_cfg.triton_model_name))
    return str(inference_cfg.triton_model_name)
