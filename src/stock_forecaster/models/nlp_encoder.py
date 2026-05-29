"""FinBERT context encoder wrapper."""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from stock_forecaster.models.interfaces import ContextEncoder


class FinBertEncoder(ContextEncoder):
    """Extract [CLS] embeddings from FinBERT and project to fusion space."""

    def __init__(
        self,
        model_name: str,
        embed_dim: int,
        max_length: int = 128,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._embed_dim = embed_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if unfreeze_last_n_layers > 0:
            encoder_layers = self.backbone.encoder.layer
            for layer in encoder_layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden_size, embed_dim)

    @property
    def output_dim(self) -> int:
        return self._embed_dim

    def forward(self, news_texts: list[str]) -> torch.Tensor:
        """
        Encode a batch of concatenated news strings.

        Args:
            news_texts: list length batch with raw text per sample.
        """
        encoded = self.tokenizer(
            news_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.backbone(**encoded)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_hidden)

    def encode_sequence_batch(self, daily_news_batch: list[list[str]]) -> torch.Tensor:
        """
        Encode per-timestep news strings.

        Args:
            daily_news_batch: batch of ``seq_len`` headline strings per sample.

        Returns:
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        """
        if not daily_news_batch:
            msg = "daily_news_batch must not be empty"
            raise ValueError(msg)
        batch_size = len(daily_news_batch)
        seq_len = len(daily_news_batch[0])
        flat_texts = [text for sample in daily_news_batch for text in sample]
        flat_embeddings = self.forward(flat_texts)
        return flat_embeddings.view(batch_size, seq_len, -1)
