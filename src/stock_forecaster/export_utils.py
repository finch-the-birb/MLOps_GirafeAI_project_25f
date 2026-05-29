"""Shared ONNX export helpers for CLI and MLflow artifact logging."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from stock_forecaster.lightning_module import StockForecaster
from stock_forecaster.models.early_fusion_model import EarlyFusionStockModel

logger = logging.getLogger(__name__)


class OnnxLateFusionWrapper(torch.nn.Module):
    """
    Late fusion ONNX: time series + single precomputed FinBERT CLS per sample.

    FinBERT tokenization runs outside Triton (FastAPI preprocess).
    """

    def __init__(self, forecaster: StockForecaster) -> None:
        super().__init__()
        self.ts_encoder = forecaster.model.ts_encoder
        self.text_proj = forecaster.model.nlp_encoder.projection
        self.fusion = forecaster.model.fusion
        self.classifier = forecaster.classifier

    def forward(
        self,
        time_series: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        ts_latent = self.ts_encoder(time_series)
        text_latent = self.text_proj(text_embedding)
        fused = self.fusion(ts_latent, text_latent)
        logits = self.classifier(fused)
        return torch.sigmoid(logits)


class OnnxEarlyFusionWrapper(torch.nn.Module):
    """
    Early fusion ONNX: OHLCV + per-step projected text features -> iTransformer.

    Preprocess must supply ``text_per_step`` of shape (batch, seq_len, text_feat_dim).
    """

    def __init__(self, forecaster: StockForecaster) -> None:
        super().__init__()
        self.ts_encoder = forecaster.model.ts_encoder
        self.classifier = forecaster.classifier
        self._multimodal_mode = getattr(
            forecaster.model.ts_encoder,
            "multimodal_mode",
            None,
        )

    def forward(
        self,
        time_series: torch.Tensor,
        text_per_step: torch.Tensor,
    ) -> torch.Tensor:
        if self._multimodal_mode is not None:
            latent = self.ts_encoder(time_series, text_feats=text_per_step)
        else:
            combined = torch.cat([time_series, text_per_step], dim=-1)
            latent = self.ts_encoder(combined)
        logits = self.classifier(latent)
        return torch.sigmoid(logits)


# Backward-compatible alias
OnnxInferenceWrapper = OnnxLateFusionWrapper


def export_forecaster_to_onnx(
    forecaster: StockForecaster,
    cfg: DictConfig,
    output_path: Path | None = None,
) -> Path:
    """Export late- or early-fusion forecaster to ONNX."""
    forecaster.eval()
    seq_len = int(cfg.data.seq_len)
    n_price_features = int(cfg.data.n_features)

    if output_path is None:
        output_path = Path(cfg.get("onnx_output", "artifacts/multimodal_model.onnx"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_batch = cfg.get("dynamic_batch", True)

    if isinstance(forecaster.model, EarlyFusionStockModel):
        text_feat_dim = int(forecaster.model.text_feat_dim)
        wrapper = OnnxEarlyFusionWrapper(forecaster)
        dummy_ts = torch.randn(1, seq_len, n_price_features)
        dummy_text = torch.randn(1, seq_len, text_feat_dim)
        input_names = ["time_series", "text_per_step"]
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "time_series": {0: "batch"},
                "text_per_step": {0: "batch"},
                "probability": {0: "batch"},
            }
        export_args = (dummy_ts, dummy_text)
    else:
        wrapper = OnnxLateFusionWrapper(forecaster)
        dummy_ts = torch.randn(1, seq_len, n_price_features)
        hidden_size = forecaster.model.nlp_encoder.backbone.config.hidden_size
        dummy_text = torch.randn(1, hidden_size)
        input_names = ["time_series", "text_embedding"]
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "time_series": {0: "batch"},
                "text_embedding": {0: "batch"},
                "probability": {0: "batch"},
            }
        export_args = (dummy_ts, dummy_text)

    torch.onnx.export(
        wrapper,
        export_args,
        str(output_path),
        input_names=input_names,
        output_names=["probability"],
        dynamic_axes=dynamic_axes,
        opset_version=int(cfg.get("opset_version", 17)),
    )
    logger.info("Exported ONNX model to %s", output_path)

    triton_dest = cfg.get("triton_copy")
    if triton_dest:
        dest = Path(triton_dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(output_path.read_bytes())
        logger.info("Copied ONNX to Triton model repo: %s", dest)

    return output_path
