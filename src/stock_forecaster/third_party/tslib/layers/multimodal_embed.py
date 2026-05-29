"""Multimodal inverted embedding for iTransformer (numeric variates + text)."""

from __future__ import annotations

import torch
from torch import nn

MULTIMODAL_MODES = frozenset({"flatten", "daily_mlp", "gated_fusion", "transformer_join"})


class MultimodalITransformerEmbedding(nn.Module):
    """
    iTransformer-style inverted embedding with four multimodal fusion strategies.

    All modes output tokens for the shared iTransformer encoder. Only ``flatten``
    appends an extra text token (7 total); the other modes fuse text into six
    numeric variate tokens (6 total).

    Expected inputs:
        x_numeric: [B, L, F_num]  (e.g. F_num=6)
        x_text:    [B, L, text_feat_dim]  (e.g. text_feat_dim=32)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_numeric_features: int = 6,
        text_feat_dim: int = 32,
        multimodal_mode: str = "flatten",
        dropout: float = 0.1,
        n_heads: int = 4,
        text_m_dim: int = 1,
    ) -> None:
        super().__init__()
        del text_m_dim  # reserved for legacy configs; unused in the four ablation modes

        if multimodal_mode not in MULTIMODAL_MODES:
            msg = (
                f"multimodal_mode must be one of {sorted(MULTIMODAL_MODES)}, "
                f"got {multimodal_mode!r}"
            )
            raise ValueError(msg)

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_numeric_features = n_numeric_features
        self.text_feat_dim = text_feat_dim
        self.multimodal_mode = multimodal_mode
        self.n_heads = n_heads

        self.numeric_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if multimodal_mode == "flatten":
            self.text_flat_embedding = nn.Linear(seq_len * text_feat_dim, d_model)
            self.daily_mlp = None
            self.text_gate = None
            self.text_to_d = None
            self.cross_attn = None
            self.cross_attn_norm = None
        elif multimodal_mode == "daily_mlp":
            self.text_flat_embedding = None
            hybrid_dim = n_numeric_features + text_feat_dim
            self.daily_mlp = nn.Sequential(
                nn.Linear(hybrid_dim, n_numeric_features),
                nn.ReLU(),
                nn.LayerNorm(n_numeric_features),
            )
            self.text_gate = None
            self.text_to_d = None
            self.cross_attn = None
            self.cross_attn_norm = None
        elif multimodal_mode == "gated_fusion":
            self.text_flat_embedding = None
            self.daily_mlp = None
            self.text_gate = nn.Sequential(
                nn.Linear(text_feat_dim, n_numeric_features),
                nn.Sigmoid(),
            )
            self.text_to_d = None
            self.cross_attn = None
            self.cross_attn_norm = None
        else:  # transformer_join
            self.text_flat_embedding = None
            self.daily_mlp = None
            self.text_gate = None
            self.text_to_d = nn.Linear(text_feat_dim, d_model)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(d_model)

    @property
    def num_tokens(self) -> int:
        if self.multimodal_mode == "flatten":
            return self.n_numeric_features + 1
        return self.n_numeric_features

    def _validate_inputs(self, x_numeric: torch.Tensor, x_text: torch.Tensor) -> None:
        if x_numeric.shape[1] != self.seq_len:
            msg = f"Expected seq_len={self.seq_len}, got {x_numeric.shape[1]} for x_numeric"
            raise ValueError(msg)
        if x_numeric.shape[2] != self.n_numeric_features:
            msg = (
                f"Expected n_numeric_features={self.n_numeric_features}, "
                f"got {x_numeric.shape[2]} for x_numeric"
            )
            raise ValueError(msg)
        if x_text.shape[1] != self.seq_len:
            msg = f"Expected seq_len={self.seq_len}, got {x_text.shape[1]} for x_text"
            raise ValueError(msg)
        if x_text.shape[2] != self.text_feat_dim:
            msg = (
                f"Expected text_feat_dim={self.text_feat_dim}, " f"got {x_text.shape[2]} for x_text"
            )
            raise ValueError(msg)

    def _embed_numeric_from_series(self, x_numeric: torch.Tensor) -> torch.Tensor:
        """[B, L, F_num] -> [B, F_num, D] via inverted variate embedding."""
        return self.numeric_embedding(x_numeric.permute(0, 2, 1))

    def _forward_flatten(self, x_numeric: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        numeric_tokens = self._embed_numeric_from_series(x_numeric)
        batch_size = x_text.size(0)
        flat_text = x_text.reshape(batch_size, self.seq_len * self.text_feat_dim)
        text_token = self.text_flat_embedding(flat_text).unsqueeze(1)
        return torch.cat([numeric_tokens, text_token], dim=1)

    def _forward_daily_mlp(self, x_numeric: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        hybrid = torch.cat([x_numeric, x_text], dim=-1)
        fused = self.daily_mlp(hybrid)
        return self._embed_numeric_from_series(fused)

    def _forward_gated_fusion(self, x_numeric: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        gates = self.text_gate(x_text)
        fused = x_numeric * gates
        return self._embed_numeric_from_series(fused)

    def _forward_transformer_join(
        self,
        x_numeric: torch.Tensor,
        x_text: torch.Tensor,
    ) -> torch.Tensor:
        numeric_tokens = self._embed_numeric_from_series(x_numeric)
        text_tokens = self.text_to_d(x_text)
        attn_out, _ = self.cross_attn(
            query=numeric_tokens,
            key=text_tokens,
            value=text_tokens,
            need_weights=False,
        )
        return self.cross_attn_norm(numeric_tokens + attn_out)

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_text: torch.Tensor,
        series_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del series_mark  # API parity with DataEmbeddingInverted
        self._validate_inputs(x_numeric, x_text)

        if self.multimodal_mode == "flatten":
            tokens = self._forward_flatten(x_numeric, x_text)
        elif self.multimodal_mode == "daily_mlp":
            tokens = self._forward_daily_mlp(x_numeric, x_text)
        elif self.multimodal_mode == "gated_fusion":
            tokens = self._forward_gated_fusion(x_numeric, x_text)
        else:
            tokens = self._forward_transformer_join(x_numeric, x_text)

        return self.dropout(tokens)


# Backward-compatible aliases
MultimodalInvertedEmbedding = MultimodalITransformerEmbedding
MultimodaliTransformerEmbedding = MultimodalITransformerEmbedding
