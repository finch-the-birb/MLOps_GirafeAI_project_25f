"""Time-series encoder backed by TSLib iTransformer."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional

from stock_forecaster.models.interfaces import TimeSeriesEncoder
from stock_forecaster.models.revin import RevIN
from stock_forecaster.third_party.tslib.layers.multimodal_embed import MULTIMODAL_MODES
from stock_forecaster.third_party.tslib.models.itransformer import Model as TSLibITransformerModel


class TSLibITransformerEncoder(TimeSeriesEncoder):
    """
    Wrapper around the official TSLib iTransformer encoder stack.

    Returns a latent vector for multimodal fusion (not class logits).
    Input: (batch, seq_len, n_features) — same layout as TSLib [B, L, N].

    When ``multimodal_mode`` is set (``flatten``, ``daily_mlp``, ``gated_fusion``,
    or ``transformer_join``), pass ``text_feats`` with shape
    [B, seq_len, text_feat_dim]; numeric OHLCV uses ``n_price_features`` only.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        factor: int = 1,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        use_revin: bool = True,
        latent_dim: int | None = None,
        pred_len: int = 1,
        multimodal_mode: str | None = None,
        text_m_dim: int = 1,
        n_price_features: int | None = None,
        text_feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.multimodal_mode = multimodal_mode
        self.text_m_dim = text_m_dim
        self.n_price_features = n_price_features if n_price_features is not None else n_features
        self.text_feat_dim = text_feat_dim

        if multimodal_mode is not None:
            if multimodal_mode not in MULTIMODAL_MODES:
                msg = (
                    f"multimodal_mode must be one of {sorted(MULTIMODAL_MODES)}, "
                    f"got {multimodal_mode!r}"
                )
                raise ValueError(msg)
            enc_in = self.n_price_features
            revin_features = self.n_price_features
        else:
            enc_in = n_features
            revin_features = n_features

        self.n_features = enc_in
        self._d_model = d_model
        self._latent_dim = latent_dim or d_model

        configs = SimpleNamespace(
            task_name="anomaly_detection",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            embed=embed,
            freq=freq,
            factor=factor,
            num_class=1,
            multimodal_mode=multimodal_mode,
            text_m_dim=text_m_dim,
            n_numeric_features=self.n_price_features,
            text_feat_dim=text_feat_dim,
        )
        self.backbone = TSLibITransformerModel(configs)
        self.revin = RevIN(revin_features) if use_revin else None
        self.post_dropout = nn.Dropout(dropout)
        self.latent_proj = nn.Linear(d_model * enc_in, self._latent_dim)

    @property
    def output_dim(self) -> int:
        return self._latent_dim

    @property
    def d_model(self) -> int:
        return self._latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode_variates(
        self,
        time_series: torch.Tensor,
        text_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run inverted embedding + encoder; return flattened numeric representation."""
        if self.multimodal_mode is not None:
            if text_feats is None:
                msg = "text_feats is required when multimodal_mode is set"
                raise ValueError(msg)
            enc_out = self.backbone.encode_multimodal(time_series, text_feats)
        else:
            enc_out = self.backbone.enc_embedding(time_series, None)
            enc_out, _ = self.backbone.encoder(enc_out, attn_mask=None)

        enc_out = functional.gelu(enc_out)
        enc_out = self.post_dropout(enc_out)
        return enc_out.reshape(enc_out.size(0), -1)

    def forward(
        self,
        time_series: torch.Tensor,
        text_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.revin is not None:
            time_series = self.revin(time_series, mode="norm")
        flattened = self.encode_variates(time_series, text_feats=text_feats)
        return self.latent_proj(flattened)


# Backward-compatible alias used in docs and earlier configs.
ITransformerEncoder = TSLibITransformerEncoder
