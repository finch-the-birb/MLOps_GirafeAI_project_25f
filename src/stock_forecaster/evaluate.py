"""Run test-set evaluation from a saved Lightning checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from stock_forecaster.data.datamodule import FnspidDataModule, positive_class_weight
from stock_forecaster.data.download_data import download_data
from stock_forecaster.lightning_module import StockForecaster
from stock_forecaster.train import _resolve_trainer_devices

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(cfg.seed)

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.is_file():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    run_label = str(cfg.get("eval_label", checkpoint_path.stem))
    logger.info("Evaluating run=%s checkpoint=%s", run_label, checkpoint_path)

    processed_path = Path(cfg.data.processed_file)
    download_data(processed_path, top_tickers=cfg.data.top_tickers)

    data_module = FnspidDataModule(cfg.data)
    data_module.setup("fit")

    pos_weight: float | None = None
    if cfg.trainer.get("use_class_weights", False):
        pos_weight = positive_class_weight(data_module.train_dataset)  # type: ignore[arg-type]

    forecaster = StockForecaster(
        model=instantiate(cfg.model),
        optimizer_cfg=cfg.trainer.optimizer,
        scheduler_cfg=cfg.trainer.get("scheduler"),
        pos_weight=pos_weight,
    )

    accelerator, devices = _resolve_trainer_devices(cfg)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=cfg.trainer.precision,
        logger=False,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
    )

    metrics_list = trainer.test(
        forecaster,
        datamodule=data_module,
        ckpt_path=str(checkpoint_path),
    )
    metrics = metrics_list[0] if metrics_list else {}

    output_path = Path(cfg.get("eval_output", f"artifacts/metrics_{run_label}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": run_label,
        "checkpoint": str(checkpoint_path),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data_fusion_mode": str(cfg.data.get("fusion_mode", "late")),
        "test_metrics": {key: float(value) for key, value in metrics.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Test metrics written to %s", output_path)

    print(f"\n=== Test metrics ({run_label}) ===")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
