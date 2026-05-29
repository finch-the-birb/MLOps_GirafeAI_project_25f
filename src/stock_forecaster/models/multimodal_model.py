"""Full multimodal encoder stack (TS + NLP + fusion)."""

import torch
from torch import nn

from stock_forecaster.models.fusion import MultimodalFusion
from stock_forecaster.models.interfaces import ContextEncoder, TimeSeriesEncoder


class MultimodalStockModel(nn.Module):
    """Composable multimodal body without classification head."""

    def __init__(
        self,
        ts_encoder: TimeSeriesEncoder,
        nlp_encoder: ContextEncoder,
        fusion: MultimodalFusion,
        fusion_dim: int,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.ts_encoder = ts_encoder
        self.nlp_encoder = nlp_encoder
        self.fusion = fusion
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

    def encode(
        self,
        time_series: torch.Tensor,
        news_texts: list[str],
    ) -> torch.Tensor:
        ts_latent = self.ts_encoder(time_series)
        text_latent = self.nlp_encoder(news_texts)
        return self.fusion(ts_latent, text_latent)

    def forward(
        self,
        time_series: torch.Tensor,
        news_texts: list[str],
    ) -> torch.Tensor:
        return self.encode(time_series, news_texts)
