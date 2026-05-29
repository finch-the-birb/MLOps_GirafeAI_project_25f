"""PyTorch Lightning DataModule for FNSPID windows."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from stock_forecaster.data.dataset import FnspidWindowDataset, load_processed_frame
from stock_forecaster.data.splits import resolve_year_splits


def positive_class_weight(dataset: FnspidWindowDataset) -> float:
    """BCE pos_weight = n_negative / n_positive on the given split."""
    labels = [int(sample.label) for sample in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


class FnspidDataModule(pl.LightningDataModule):
    """Train/val/test dataloaders with temporal year splits."""

    def __init__(self, data_cfg: DictConfig) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.train_dataset: FnspidWindowDataset | None = None
        self.val_dataset: FnspidWindowDataset | None = None
        self.test_dataset: FnspidWindowDataset | None = None
        self.fold_name: str | None = None
        self.train_years: list[int] = []
        self.val_years: list[int] = []
        self.test_years: list[int] = []

    def setup(self, stage: str | None = None) -> None:
        frame = load_processed_frame(Path(self.data_cfg.processed_file))
        train_years, val_years, test_years, fold_name = resolve_year_splits(self.data_cfg)
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.fold_name = fold_name

        dataset_kwargs = {
            "seq_len": self.data_cfg.seq_len,
            "feature_columns": list(self.data_cfg.feature_columns),
            "max_news_per_window": self.data_cfg.max_news_per_window,
            "max_news_chars": self.data_cfg.max_news_chars,
            "fusion_mode": str(self.data_cfg.get("fusion_mode", "late")),
            "max_news_chars_per_day": int(self.data_cfg.get("max_news_chars_per_day", 256)),
            "window_stride": int(self.data_cfg.get("window_stride", 1)),
        }
        self.train_dataset = FnspidWindowDataset(
            frame,
            years=train_years,
            **dataset_kwargs,
        )
        self.val_dataset = FnspidWindowDataset(
            frame,
            years=val_years,
            **dataset_kwargs,
        )
        self.test_dataset = FnspidWindowDataset(
            frame,
            years=test_years,
            **dataset_kwargs,
        )

    def _collate(self, batch: list[dict[str, object]]) -> dict[str, object]:
        time_series = torch.stack([item["time_series"] for item in batch])  # type: ignore[arg-type]
        labels = torch.stack([item["label"] for item in batch])  # type: ignore[arg-type]
        fusion_mode = str(self.data_cfg.get("fusion_mode", "late"))
        if fusion_mode == "early":
            daily_news = [list(item["daily_news"]) for item in batch]  # type: ignore[arg-type]
            return {
                "time_series": time_series,
                "news_text": daily_news,
                "label": labels,
            }
        news_texts = [str(item["news_text"]) for item in batch]
        return {
            "time_series": time_series,
            "news_text": news_texts,
            "label": labels,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            collate_fn=self._collate,
        )
