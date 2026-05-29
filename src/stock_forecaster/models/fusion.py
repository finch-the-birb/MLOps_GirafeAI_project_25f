"""Cross-modal fusion between time-series and text latents."""

import torch
from torch import nn


class MultimodalFusion(nn.Module):
    """
    Fuse TS and text representations via cross-attention.

    Returns a dense state vector suitable for downstream RL agents.
    """

    def __init__(
        self,
        ts_dim: int,
        text_dim: int,
        fusion_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ts_proj = nn.Linear(ts_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.ffn[-1].out_features  # type: ignore[index]

    def forward(
        self,
        ts_latent: torch.Tensor,
        text_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ts_latent: (batch, ts_dim)
            text_latent: (batch, text_dim)
        Returns:
            fused state: (batch, fusion_dim)
        """
        ts_token = self.ts_proj(ts_latent).unsqueeze(1)
        text_token = self.text_proj(text_latent).unsqueeze(1)
        # TS queries text (cross-modal attention)
        attended, _ = self.cross_attn(ts_token, text_token, text_token)
        fused = self.norm(ts_token + attended)
        pooled = fused.squeeze(1)
        return self.ffn(pooled) + pooled
