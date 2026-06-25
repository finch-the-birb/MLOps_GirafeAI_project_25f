"""Early-fusion multimodal model: per-day FinBERT features inside iTransformer."""

from __future__ import annotations

import torch
from torch import nn

from stock_forecaster.models.interfaces import ContextEncoder, TimeSeriesEncoder


class EarlyFusionStockModel(nn.Module):
    """Encode OHLCV + daily news through a shared iTransformer stack."""

    def __init__(
        self,
        ts_encoder: TimeSeriesEncoder,
        nlp_encoder: ContextEncoder,
        classifier_input_dim: int,
        num_classes: int = 1,
        n_price_features: int = 6,
        text_feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.ts_encoder = ts_encoder
        self.nlp_encoder = nlp_encoder
        self.classifier_input_dim = classifier_input_dim
        self.num_classes = num_classes
        self.n_price_features = n_price_features
        self.text_feat_dim = text_feat_dim

    def encode(
        self,
        time_series: torch.Tensor,
        news_text: list[str] | list[list[str]],
    ) -> torch.Tensor:
        if not news_text:
            msg = "news_text must not be empty"
            raise ValueError(msg)
        if isinstance(news_text[0], str):
            msg = "EarlyFusionStockModel expects per-day news: list[list[str]]"
            raise TypeError(msg)

        text_feats = self.nlp_encoder.encode_sequence_batch(news_text)  # type: ignore[union-attr]
        price_series = time_series[:, :, : self.n_price_features]
        return self.ts_encoder(price_series, text_feats=text_feats)

    def forward(
        self,
        time_series: torch.Tensor,
        news_text: list[str] | list[list[str]],
    ) -> torch.Tensor:
        return self.encode(time_series, news_text)
