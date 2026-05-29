"""Embedding layers (from TSLib layers/Embed.py)."""

import torch
from torch import nn


class DataEmbeddingInverted(nn.Module):
    """Inverted embedding: variates as tokens (iTransformer)."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, series: torch.Tensor, series_mark: torch.Tensor | None) -> torch.Tensor:
        # series: [Batch, Time, Variate]
        values = series.permute(0, 2, 1)
        if series_mark is None:
            embedded = self.value_embedding(values)
        else:
            embedded = self.value_embedding(
                torch.cat([values, series_mark.permute(0, 2, 1)], dim=1)
            )
        return self.dropout(embedded)
