"""Export trained multimodal classifier to ONNX for Triton deployment."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from stock_forecaster.export_utils import export_forecaster_to_onnx
from stock_forecaster.lightning_module import StockForecaster

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    multimodal_model = instantiate(cfg.model)
    forecaster = StockForecaster(
        model=multimodal_model,
        optimizer_cfg=cfg.trainer.optimizer,
        scheduler_cfg=cfg.trainer.get("scheduler"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    forecaster.load_state_dict(checkpoint["state_dict"])
    export_forecaster_to_onnx(forecaster, cfg)


if __name__ == "__main__":
    main()
