"""Evaluate checkpoint on every dense inference-style test window (stride=1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stock_forecaster.data.datamodule import FnspidDataModule, positive_class_weight
from stock_forecaster.data.inference_window import iter_dense_test_samples
from stock_forecaster.lightning_module import StockForecaster

logger = logging.getLogger(__name__)


class _DenseSampleDataset(Dataset):
    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._samples[index].record
        return {
            "time_series": torch.from_numpy(record.time_series),
            "daily_news": list(record.daily_news),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "ticker": record.ticker,
            "target_date": record.target_date,
        }


def _collate_early(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "time_series": torch.stack([item["time_series"] for item in batch]),  # type: ignore[arg-type]
        "news_text": [list(item["daily_news"]) for item in batch],
        "label": torch.stack([item["label"] for item in batch]),  # type: ignore[arg-type]
    }


def _compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "val_accuracy": float(accuracy_score(labels, preds)),
        "val_precision": float(precision_score(labels, preds, zero_division=0)),
        "val_recall": float(recall_score(labels, preds, zero_division=0)),
        "val_f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["val_roc_auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["val_roc_auc"] = float("nan")
    logits = np.log(probs / np.clip(1.0 - probs, 1e-6, None))
    loss = nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    metrics["val_loss"] = float(loss.item())
    return metrics


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="eval/dense",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.is_file():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    run_label = str(cfg.get("eval_label", "dense_test"))
    logger.info("Dense inference-style eval: %s", run_label)

    dense_samples = iter_dense_test_samples(cfg.data)
    logger.info(
        "Built %d dense test samples (all trading days, inference windows)", len(dense_samples)
    )

    pos_weight: float | None = None
    if cfg.trainer.get("use_class_weights", False):
        dm = FnspidDataModule(cfg.data)
        dm.setup("fit")
        pos_weight = positive_class_weight(dm.train_dataset)  # type: ignore[arg-type]

    forecaster = StockForecaster(
        model=instantiate(cfg.model),
        optimizer_cfg=cfg.trainer.optimizer,
        scheduler_cfg=cfg.trainer.get("scheduler"),
        pos_weight=pos_weight,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    forecaster.load_state_dict(checkpoint["state_dict"])
    forecaster.to(device)
    forecaster.eval()

    dataset = _DenseSampleDataset(dense_samples)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        collate_fn=_collate_early,
    )

    all_probs: list[float] = []
    all_labels: list[int] = []
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            time_series = batch["time_series"].to(device)
            news_text = batch["news_text"]
            labels = batch["label"]
            logits = forecaster(time_series, news_text)  # type: ignore[arg-type]
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.reshape(-1).tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
            if batch_idx in (1, total_batches) or batch_idx % 50 == 0:
                logger.info(
                    "Dense inference progress: batch %d/%d (%.1f%%)",
                    batch_idx,
                    total_batches,
                    100.0 * batch_idx / total_batches,
                )

    labels_arr = np.array(all_labels, dtype=int)
    probs_arr = np.array(all_probs, dtype=float)
    metrics = _compute_metrics(labels_arr, probs_arr)

    # Stride-30 reference from standard test dataloader
    stride_metrics: dict[str, float] | None = None
    if cfg.get("include_stride_reference", True):
        import pytorch_lightning as pl

        from stock_forecaster.train import _resolve_trainer_devices

        data_module = FnspidDataModule(cfg.data)
        data_module.setup("fit")
        accelerator, devices = _resolve_trainer_devices(cfg)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=cfg.trainer.precision,
            logger=False,
            enable_progress_bar=False,
        )
        ref_list = trainer.test(forecaster, datamodule=data_module, ckpt_path=str(checkpoint_path))
        if ref_list:
            stride_metrics = {k: float(v) for k, v in ref_list[0].items()}

    output_path = Path(cfg.get("eval_output", f"artifacts/metrics_{run_label}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": run_label,
        "eval_mode": "dense_inference_windows",
        "checkpoint": str(checkpoint_path),
        "n_samples": len(dense_samples),
        "window_stride_train": int(cfg.data.get("window_stride", 30)),
        "test_metrics_dense": metrics,
        "test_metrics_stride_reference": stride_metrics,
        "data": {
            "processed_file": str(cfg.data.processed_file),
            "seq_len": int(cfg.data.seq_len),
            "label_threshold_pct": float(cfg.data.get("label_threshold_pct", 0.3)),
            "fusion_mode": str(cfg.data.get("fusion_mode", "early")),
        },
        "model": OmegaConf.to_container(cfg.model, resolve=True),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)

    print(f"\n=== Dense test metrics ({run_label}, n={len(dense_samples)}) ===")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6f}")
    if stride_metrics:
        print("\n=== Stride reference (training dataloader) ===")
        for key in sorted(stride_metrics):
            print(f"  {key}: {stride_metrics[key]:.6f}")


if __name__ == "__main__":
    main()
