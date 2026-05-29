"""Transformer encoder (from TSLib layers/Transformer_EncDec.py, trimmed)."""

import torch
from torch import nn
from torch.nn import functional


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = functional.relu if activation == "relu" else functional.gelu

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        new_hidden, attn = self.attention(
            hidden, hidden, hidden, attn_mask=attn_mask, tau=tau, delta=delta
        )
        hidden = hidden + self.dropout(new_hidden)
        residual = hidden = self.norm1(hidden)
        hidden = self.dropout(self.activation(self.conv1(residual.transpose(-1, 1))))
        hidden = self.dropout(self.conv2(hidden).transpose(-1, 1))
        return self.norm2(residual + hidden), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers: list[nn.Module], norm_layer: nn.Module | None = None) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        attentions: list[torch.Tensor | None] = []
        for attn_layer in self.attn_layers:
            hidden, attn = attn_layer(hidden, attn_mask=attn_mask, tau=tau, delta=delta)
            attentions.append(attn)
        if self.norm is not None:
            hidden = self.norm(hidden)
        return hidden, attentions
