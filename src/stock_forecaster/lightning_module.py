"""PyTorch Lightning module for supervised stock direction classification."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class StockForecaster(LightningModule):
    """Wraps late- or early-fusion multimodal models with a classification head."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig | None = None,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.pos_weight_tensor: torch.Tensor | None = None
        if pos_weight is not None:
            self.pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
        head_dim = getattr(model, "classifier_input_dim", None)
        if head_dim is None:
            head_dim = model.fusion_dim
        self.classifier = nn.Linear(head_dim, model.num_classes)
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.val_roc_auc = BinaryAUROC()

    def forward(
        self,
        time_series: torch.Tensor,
        news_text: list[str] | list[list[str]],
    ) -> torch.Tensor:
        state = self.model.encode(time_series, news_text)
        logits = self.classifier(state)
        return logits.squeeze(-1)

    def _shared_step(self, batch: dict[str, object], stage: str) -> torch.Tensor:
        time_series = batch["time_series"]
        news_text = batch["news_text"]
        labels = batch["label"]
        logits = self.forward(time_series, news_text)  # type: ignore[arg-type]
        pos_weight = self.pos_weight_tensor
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)
        loss = functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
        )
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        batch_size = int(labels.shape[0])

        if stage == "train":
            self.train_accuracy(preds, labels.int())
            self.train_precision(preds, labels.int())
            self.train_f1(preds, labels.int())
            self.train_recall(preds, labels.int())
        elif stage == "val":
            self.val_accuracy(preds, labels.int())
            self.val_precision(preds, labels.int())
            self.val_f1(preds, labels.int())
            self.val_recall(preds, labels.int())
            self.val_roc_auc(probs, labels.int())
        self._last_batch_size = batch_size
        return loss

    def training_step(self, batch: dict[str, object], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        batch_size = self._last_batch_size
        self.log(
            "train_loss",
            loss.detach().float(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, batch_size=batch_size)
        self.log("train_precision", self.train_precision, on_epoch=True, batch_size=batch_size)
        self.log("train_f1", self.train_f1, on_epoch=True, batch_size=batch_size)
        self.log("train_recall", self.train_recall, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict[str, object], _batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        batch_size = self._last_batch_size
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, batch_size=batch_size)
        self.log("val_precision", self.val_precision, on_epoch=True, batch_size=batch_size)
        self.log("val_f1", self.val_f1, on_epoch=True, batch_size=batch_size)
        self.log("val_recall", self.val_recall, on_epoch=True, batch_size=batch_size)
        self.log("val_roc_auc", self.val_roc_auc, on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, batch: dict[str, object], _batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, _batch_idx)

    def configure_optimizers(self):
        base_lr = float(self.optimizer_cfg.lr)
        weight_decay = float(self.optimizer_cfg.weight_decay)
        nlp_lr = self.optimizer_cfg.get("nlp_lr")
        nlp_encoder = getattr(self.model, "nlp_encoder", None)
        nlp_param_ids: set[int] = set()
        if nlp_encoder is not None:
            nlp_param_ids = {id(param) for param in nlp_encoder.parameters() if param.requires_grad}

        nlp_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) in nlp_param_ids
        ]
        other_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in nlp_param_ids
        ]

        if nlp_params and nlp_lr is not None:
            param_groups = [
                {"params": nlp_params, "lr": float(nlp_lr), "weight_decay": weight_decay},
                {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
            ]
        else:
            param_groups = [
                {"params": nlp_params + other_params, "lr": base_lr, "weight_decay": weight_decay}
            ]

        optimizer = torch.optim.AdamW(param_groups)
        if self.scheduler_cfg is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_cfg.T_max,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
