"""CLI inference: historical window from FNSPID parquet → Triton."""

from __future__ import annotations

import json
import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from stock_forecaster.service.fusion import resolve_fusion_mode, triton_model_name
from stock_forecaster.service.historical import build_historical_inference_payload
from stock_forecaster.service.triton_client import predict_triton

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    inference_cfg = cfg.inference
    ticker = str(inference_cfg.ticker).upper()
    target_date = str(inference_cfg.target_date)

    fusion_mode = resolve_fusion_mode(None, cfg.data, inference_cfg)
    model_name = triton_model_name(inference_cfg, fusion_mode)

    nlp_encoder = instantiate(cfg.model.nlp_encoder)
    nlp_encoder.eval()
    nlp_encoder.to(torch.device("cpu"))

    payload = build_historical_inference_payload(
        ticker=ticker,
        target_date=target_date,
        data_cfg=cfg.data,
        nlp_encoder=nlp_encoder,
        fusion_mode=fusion_mode,
    )
    probability = predict_triton(
        triton_url=inference_cfg.triton_url,
        model_name=model_name,
        model_version=inference_cfg.triton_model_version,
        time_series=payload.time_series,
        fusion_mode=fusion_mode,
        text_embedding=payload.text_embedding,
        text_per_step=payload.text_per_step,
    )
    result = {
        "ticker": ticker,
        "target_date": target_date,
        "fusion_mode": fusion_mode,
        "probability_up": probability,
        "predicted_label": int(probability >= 0.5),
        "news_analyzed": payload.news_analyzed,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
