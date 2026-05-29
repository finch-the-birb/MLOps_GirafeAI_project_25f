"""FastAPI HTTP service: preprocess inputs and query Triton."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from stock_forecaster.service.preprocess import build_inference_payload
from stock_forecaster.service.routes_v1 import register_v1_routes
from stock_forecaster.service.routes_v1 import router as v1_router
from stock_forecaster.service.triton_client import predict_triton

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])


class PredictResponse(BaseModel):
    ticker: str
    probability_up: float
    predicted_label: int


def _load_cfg() -> DictConfig:
    conf_dir = Path(__file__).resolve().parents[3] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(config_name="config")


@lru_cache(maxsize=1)
def _get_nlp_encoder(cfg: DictConfig):
    encoder = instantiate(cfg.model.nlp_encoder)
    encoder.eval()
    return encoder


def create_app(cfg: DictConfig | None = None) -> FastAPI:
    app = FastAPI(title="Stock Forecaster API", version="0.1.0")
    runtime_cfg = cfg or _load_cfg()
    inference_cfg = runtime_cfg.inference

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            nlp_encoder = _get_nlp_encoder(runtime_cfg)
            device = torch.device("cpu")
            nlp_encoder.to(device)
            time_series, text_embedding = build_inference_payload(
                ticker=request.ticker.upper(),
                data_cfg=runtime_cfg.data,
                nlp_encoder=nlp_encoder,
            )
            probability = predict_triton(
                triton_url=inference_cfg.triton_url,
                model_name=inference_cfg.triton_model_name,
                model_version=inference_cfg.triton_model_version,
                time_series=time_series,
                text_embedding=text_embedding,
            )
            label = int(probability >= 0.5)
            return PredictResponse(
                ticker=request.ticker.upper(),
                probability_up=probability,
                predicted_label=label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    register_v1_routes(v1_router, runtime_cfg=runtime_cfg, get_nlp_encoder=_get_nlp_encoder)
    app.include_router(v1_router)

    return app


app = create_app()


def main() -> None:
    cfg = _load_cfg()
    uvicorn.run(
        "stock_forecaster.service.app:app",
        host=cfg.inference.service_host,
        port=cfg.inference.service_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
