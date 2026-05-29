"""Hydra-driven training entry point with MLflow logging."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import suppress
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from stock_forecaster.data.datamodule import FnspidDataModule, positive_class_weight
from stock_forecaster.data.download_data import download_data
from stock_forecaster.data.splits import (
    folds_enabled,
    get_fold_definitions,
    iter_fold_indices,
    materialize_fold_data_cfg,
)
from stock_forecaster.export_utils import export_forecaster_to_onnx
from stock_forecaster.lightning_module import StockForecaster
from stock_forecaster.utils.git_info import get_git_commit_id
from stock_forecaster.utils.plotting import save_metric_plots

logger = logging.getLogger(__name__)


def _configure_windows_console_utf8() -> None:
    """Avoid UnicodeEncodeError when MLflow/Lightning log emoji on cp1251 consoles."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def _resolve_trainer_devices(cfg: DictConfig) -> tuple[str, int]:
    """Use GPU when CUDA is available; fall back to CPU with a clear warning."""

    requested = str(cfg.trainer.accelerator)

    device_count = int(cfg.trainer.devices)

    if requested in {"auto", "gpu", "cuda"}:
        if torch.cuda.is_available():
            return "gpu", device_count

        logger.warning(
            "CUDA is not available (installed torch: %s). "
            "Reinstall GPU PyTorch: uv sync --reinstall-package torch. Falling back to CPU.",
            torch.__version__,
        )

        return "cpu", 1

    if requested == "cpu":
        return "cpu", 1

    return requested, device_count


def _finalize_mlflow_run(
    cfg: DictConfig,
    mlflow_logger: MLFlowLogger,
    forecaster: StockForecaster,
    checkpoint_callback: ModelCheckpoint,
    plots_dir: Path,
    git_commit: str,
) -> None:
    """Log plots, checkpoints, and ONNX into the Lightning MLflow run (no second run)."""

    run_id = mlflow_logger.run_id

    if run_id is None:
        logger.warning("MLflow run_id is missing; skipping artifact logging.")

        return

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("git_commit", git_commit)

        if cfg.mlflow.log_plots:
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), artifact_path="plots")

        if not cfg.mlflow.get("log_model", False):
            return

        try:
            ckpt_path = Path(
                checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or ""
            )

            if ckpt_path.is_file():
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

            mlflow.pytorch.log_model(forecaster, artifact_path="pytorch_model")

            onnx_cfg = OmegaConf.create(
                {
                    "data": OmegaConf.to_container(cfg.data, resolve=True),
                    "onnx_output": str(Path(cfg.paths.onnx_output)),
                    "opset_version": 17,
                    "dynamic_batch": True,
                    "triton_copy": None,
                }
            )
            onnx_path = export_forecaster_to_onnx(forecaster, onnx_cfg)
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        except Exception:
            logger.exception(
                "MLflow model/ONNX artifact logging failed; "
                "training metrics and local checkpoints are kept."
            )


def _load_checkpoint_weights(forecaster: StockForecaster, ckpt_path: str | Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    forecaster.load_state_dict(checkpoint["state_dict"])


def _build_callbacks(cfg: DictConfig, checkpoint_callback: ModelCheckpoint) -> list[pl.Callback]:
    metric_history = MetricHistoryCallback()

    callbacks: list[pl.Callback] = [checkpoint_callback, metric_history]

    es_cfg = cfg.trainer.get("early_stopping")

    if es_cfg and es_cfg.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=str(es_cfg.monitor),
                patience=int(es_cfg.patience),
                mode=str(es_cfg.mode),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                verbose=True,
            )
        )

        logger.info(
            "EarlyStopping enabled: monitor=%s patience=%d mode=%s",
            es_cfg.monitor,
            es_cfg.patience,
            es_cfg.mode,
        )

    return callbacks


class MetricHistoryCallback(pl.Callback):
    """Collect epoch metrics for offline plots."""

    def __init__(self) -> None:
        super().__init__()

        self.history: dict[str, list[float]] = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []

            self.history[key].append(float(value.detach().cpu()))


def _checkpoint_dir_for_fold(cfg: DictConfig, fold_index: int, fold_name: str | None) -> Path:
    base = Path(cfg.paths.checkpoint_dir)

    if fold_name is None:
        return base

    return base / f"{fold_index:02d}_{fold_name}"


def _run_name_for_fold(cfg: DictConfig, fold_name: str | None) -> str | None:
    base = cfg.get("run_name")

    if fold_name is None:
        return str(base) if base is not None else None

    suffix = fold_name

    if base is None:
        return suffix

    return f"{base}_{suffix}"


def _train_single_fold(cfg: DictConfig, data_cfg: DictConfig, fold_index: int) -> dict[str, object]:
    data_module = FnspidDataModule(data_cfg)

    data_module.setup("fit")

    fold_name = data_module.fold_name

    logger.info(
        "Fold %s | train_years=%s val_years=%s test_years=%s | window_stride=%s | "
        "train_samples=%d val_samples=%d",
        fold_name or "default",
        data_module.train_years,
        data_module.val_years,
        data_module.test_years,
        int(data_cfg.get("window_stride", 1)),
        len(data_module.train_dataset),  # type: ignore[arg-type]
        len(data_module.val_dataset),  # type: ignore[arg-type]
    )

    pos_weight: float | None = None

    if cfg.trainer.get("use_class_weights", False):
        pos_weight = positive_class_weight(data_module.train_dataset)  # type: ignore[arg-type]

        logger.info("Train class pos_weight (neg/pos): %.4f", pos_weight)

    multimodal_model = instantiate(cfg.model)

    forecaster = StockForecaster(
        model=multimodal_model,
        optimizer_cfg=cfg.trainer.optimizer,
        scheduler_cfg=cfg.trainer.get("scheduler"),
        pos_weight=pos_weight,
    )

    checkpoint_dir = _checkpoint_dir_for_fold(cfg, fold_index, fold_name)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cfg = cfg.trainer.get("checkpoint") or {}
    monitor = str(ckpt_cfg.get("monitor", "val_loss"))
    mode = str(ckpt_cfg.get("mode", "min"))
    filename = str(ckpt_cfg.get("filename", f"best-{{epoch:02d}}-{{{monitor}:.4f}}"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=filename,
        save_top_k=1,
        save_last=True,
        monitor=monitor,
        mode=mode,
    )

    callbacks = _build_callbacks(cfg, checkpoint_callback)

    metric_history = next(c for c in callbacks if isinstance(c, MetricHistoryCallback))

    git_commit = get_git_commit_id()

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=_run_name_for_fold(cfg, fold_name),
        tags={"git_commit": git_commit, "fold": fold_name or "default"},
    )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    accelerator, devices = _resolve_trainer_devices(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(checkpoint_dir),
    )

    trainer.fit(forecaster, datamodule=data_module)

    best_ckpt = checkpoint_callback.best_model_path

    if best_ckpt:
        logger.info("Best checkpoint (%s): %s", monitor, best_ckpt)

        test_metrics_list = trainer.test(forecaster, datamodule=data_module, ckpt_path=best_ckpt)

        _load_checkpoint_weights(forecaster, best_ckpt)

    else:
        logger.warning("No best checkpoint found; testing and exporting last epoch weights.")

        test_metrics_list = trainer.test(forecaster, datamodule=data_module)

    test_metrics = test_metrics_list[0] if test_metrics_list else {}

    plots_base = Path(cfg.paths.plots_dir)
    plots_dir = plots_base / f"{fold_index:02d}_{fold_name}" if fold_name else plots_base
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_metric_plots(metric_history.history, plots_dir)

    _finalize_mlflow_run(
        cfg,
        mlflow_logger,
        forecaster,
        checkpoint_callback,
        plots_dir,
        git_commit,
    )

    return {
        "fold_index": fold_index,
        "fold_name": fold_name,
        "train_years": data_module.train_years,
        "val_years": data_module.val_years,
        "test_years": data_module.test_years,
        "train_samples": len(data_module.train_dataset),  # type: ignore[arg-type]
        "val_samples": len(data_module.val_dataset),  # type: ignore[arg-type]
        "best_checkpoint": best_ckpt,
        "mlflow_run_id": mlflow_logger.run_id,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _configure_windows_console_utf8()
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(cfg.seed)

    processed_path = Path(cfg.data.processed_file)

    # Important: label_threshold_pct affects the target_label in the processed parquet.
    # For sweeps that vary the label threshold, we must rebuild into a threshold-specific file.
    label_threshold_pct = float(cfg.data.get("label_threshold_pct", 0.5))
    force_rebuild = bool(cfg.data.get("force_rebuild", False))
    download_data(
        processed_path,
        top_tickers=cfg.data.top_tickers,
        label_threshold_pct=label_threshold_pct,
        force_rebuild=force_rebuild,
    )

    fold_indices = iter_fold_indices(cfg.data)

    if folds_enabled(cfg.data):
        logger.info(
            "Walk-forward folds: %d definition(s), running indices %s",
            len(get_fold_definitions(cfg.data)),
            fold_indices,
        )

    fold_results: list[dict[str, object]] = []

    for fold_index in fold_indices:
        data_cfg = materialize_fold_data_cfg(cfg.data, fold_index)

        fold_results.append(_train_single_fold(cfg, data_cfg, fold_index))

    if len(fold_results) > 1:
        summary_path = Path(cfg.paths.checkpoint_dir) / "fold_summary.json"

        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_path.write_text(json.dumps(fold_results, indent=2), encoding="utf-8")

        logger.info("Fold summary written to %s", summary_path)

    period_name = cfg.get("period_name")
    if period_name and len(fold_results) == 1 and "period_ablation" in str(cfg.get("run_name", "")):
        period_payload = {
            "period_name": period_name,
            "period_description": cfg.get("period_description"),
            "train_years": fold_results[0].get("train_years"),
            "val_years": fold_results[0].get("val_years"),
            "test_years": fold_results[0].get("test_years"),
            "train_samples": fold_results[0].get("train_samples"),
            "val_samples": fold_results[0].get("val_samples"),
            "checkpoint": fold_results[0].get("best_checkpoint"),
            "mlflow_run_id": fold_results[0].get("mlflow_run_id"),
            "test_metrics": fold_results[0].get("test_metrics"),
        }
        ablation_dir = Path("artifacts/period_ablation")
        ablation_dir.mkdir(parents=True, exist_ok=True)
        period_path = ablation_dir / f"{period_name}.json"
        period_path.write_text(json.dumps(period_payload, indent=2), encoding="utf-8")
        logger.info("Period ablation result written to %s", period_path)

    sweep_id = cfg.get("sweep_id")
    if (
        sweep_id
        and len(fold_results) == 1
        and (
            "hyperparam_sweep" in str(cfg.get("run_name", ""))
            or "flatten_sweep" in str(cfg.get("run_name", ""))
            or "preproj_sweep" in str(cfg.get("run_name", ""))
            or "fusion_sweep" in str(cfg.get("run_name", ""))
            or "fusion_tune_" in str(cfg.get("run_name", ""))
            or "gated_night" in str(cfg.get("run_name", ""))
            or "gated_hybrid" in str(cfg.get("run_name", ""))
        )
    ):
        ts_cfg = cfg.model.get("ts_encoder") or {}
        sweep_payload = {
            "sweep_id": str(sweep_id),
            "sweep_phase": cfg.get("sweep_phase"),
            "sweep_hypothesis": cfg.get("sweep_hypothesis"),
            "multimodal_mode": ts_cfg.get("multimodal_mode"),
            "classifier_input_dim": cfg.model.get("classifier_input_dim"),
            "train_years": fold_results[0].get("train_years"),
            "val_years": fold_results[0].get("val_years"),
            "test_years": fold_results[0].get("test_years"),
            "train_samples": fold_results[0].get("train_samples"),
            "val_samples": fold_results[0].get("val_samples"),
            "data_seq_len": int(cfg.data.seq_len),
            "data_window_stride": int(cfg.data.get("window_stride", 1)),
            "checkpoint_monitor": str(
                (cfg.trainer.get("checkpoint") or {}).get("monitor", "val_loss")
            ),
            "checkpoint": fold_results[0].get("best_checkpoint"),
            "mlflow_run_id": fold_results[0].get("mlflow_run_id"),
            "test_metrics": fold_results[0].get("test_metrics"),
        }
        sweep_dir = Path("artifacts/hyperparam_sweep")
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = sweep_dir / f"{sweep_id}.json"
        sweep_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")
        logger.info("Hyperparam sweep result written to %s", sweep_path)

    last = fold_results[-1]

    logger.info(
        "Training complete. Last fold=%s | MLflow run: %s",
        last.get("fold_name", "default"),
        last.get("mlflow_run_id"),
    )


if __name__ == "__main__":
    main()
