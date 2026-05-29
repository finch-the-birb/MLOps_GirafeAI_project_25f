"""Lightweight CLI inference entry point (Triton + optional local preprocess)."""

from __future__ import annotations

import json
import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from stock_forecaster.service.preprocess import build_inference_payload
from stock_forecaster.service.triton_client import predict_triton

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    inference_cfg = cfg.inference
    ticker = inference_cfg.ticker

    nlp_encoder = instantiate(cfg.model.nlp_encoder)
    nlp_encoder.eval()
    nlp_encoder.to(torch.device("cpu"))

    time_series, text_embedding = build_inference_payload(
        ticker=str(ticker).upper(),
        data_cfg=cfg.data,
        nlp_encoder=nlp_encoder,
    )
    probability = predict_triton(
        triton_url=inference_cfg.triton_url,
        model_name=inference_cfg.triton_model_name,
        model_version=inference_cfg.triton_model_version,
        time_series=time_series,
        text_embedding=text_embedding,
    )
    result = {
        "ticker": ticker,
        "probability_up": probability,
        "predicted_label": int(probability >= 0.5),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
