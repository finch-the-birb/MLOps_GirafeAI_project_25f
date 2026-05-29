"""Versioned HTTP routes for external clients (e.g. Streamlit UI)."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from omegaconf import DictConfig
from pydantic import BaseModel, Field, field_validator

from stock_forecaster.service.fusion import (
    FusionMode,
    resolve_fusion_mode,
    triton_model_name,
    validate_model_fusion_compatibility,
)
from stock_forecaster.service.historical import (
    build_historical_inference_payload,
    validate_target_date_2023,
)
from stock_forecaster.service.triton_client import predict_triton

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


class PredictV1Request(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    target_date: str = Field(..., examples=["2023-10-15"])
    fusion_mode: Literal["late", "early"] | None = Field(
        default=None,
        description="Override fusion mode; defaults to inference/data config",
    )

    @field_validator("target_date")
    @classmethod
    def _validate_date(cls, value: str) -> str:
        return validate_target_date_2023(value).isoformat()

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            msg = "ticker must not be empty"
            raise ValueError(msg)
        return normalized


class PredictV1Response(BaseModel):
    prediction: Literal["UP", "DOWN"]
    probability: float = Field(ge=0.0, le=1.0)
    news_analyzed: int = Field(ge=0)
    fusion_mode: FusionMode
    window_start_date: str | None = None
    window_end_date: str | None = None
    label_threshold_pct: float | None = None
    actual_forward_return_pct: float | None = None
    actual_direction: Literal["UP", "DOWN"] | None = None


def register_v1_routes(
    router: APIRouter,
    *,
    runtime_cfg: DictConfig,
    get_nlp_encoder,
) -> None:
    """Attach predict handler (factory keeps Hydra config injectable in tests)."""
    inference_cfg = runtime_cfg.inference

    @router.post("/predict", response_model=PredictV1Response)
    def predict_v1(request: PredictV1Request) -> PredictV1Response:
        try:
            import torch

            fusion_mode = resolve_fusion_mode(
                request.fusion_mode,
                runtime_cfg.data,
                inference_cfg,
            )
            validate_model_fusion_compatibility(fusion_mode, runtime_cfg.model)

            nlp_encoder = get_nlp_encoder(runtime_cfg)
            device = torch.device("cpu")
            nlp_encoder.to(device)
            payload = build_historical_inference_payload(
                ticker=request.ticker,
                target_date=request.target_date,
                data_cfg=runtime_cfg.data,
                nlp_encoder=nlp_encoder,
                fusion_mode=fusion_mode,
            )
            model_name = triton_model_name(inference_cfg, fusion_mode)
            probability_up = predict_triton(
                triton_url=inference_cfg.triton_url,
                model_name=model_name,
                model_version=inference_cfg.triton_model_version,
                time_series=payload.time_series,
                fusion_mode=fusion_mode,
                text_embedding=payload.text_embedding,
                text_per_step=payload.text_per_step,
            )
            prediction: Literal["UP", "DOWN"] = "UP" if probability_up >= 0.5 else "DOWN"
            outcome = payload.outcome
            return PredictV1Response(
                prediction=prediction,
                probability=float(probability_up),
                news_analyzed=payload.news_analyzed,
                fusion_mode=fusion_mode,
                window_start_date=payload.window_start_date,
                window_end_date=payload.window_end_date,
                label_threshold_pct=outcome.threshold_pct if outcome else None,
                actual_forward_return_pct=outcome.forward_return_pct if outcome else None,
                actual_direction=outcome.direction if outcome else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("v1 prediction failed for %s %s", request.ticker, request.target_date)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
