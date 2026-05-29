"""Build inference payloads from FNSPID test data (dynamic windows)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from stock_forecaster.data.dataset import SampleRecord
from stock_forecaster.data.inference_window import (
    ForwardReturnOutcome,
    build_dynamic_inference_sample,
    validate_target_date_2023,
)
from stock_forecaster.models.nlp_encoder import FinBertEncoder
from stock_forecaster.service.fusion import FusionMode
from stock_forecaster.service.preprocess import encode_early_per_step, encode_news


@dataclass(frozen=True, slots=True)
class HistoricalInferencePayload:
    time_series: np.ndarray
    fusion_mode: FusionMode
    news_analyzed: int
    text_embedding: np.ndarray | None = None
    text_per_step: np.ndarray | None = None
    outcome: ForwardReturnOutcome | None = None
    window_start_date: str | None = None
    window_end_date: str | None = None


def find_test_sample(
    ticker: str,
    target_date: str,
    data_cfg: DictConfig,
) -> tuple[SampleRecord, int, ForwardReturnOutcome]:
    """Build an aligned sample on the fly (any 2023 trading day with enough history)."""
    dynamic = build_dynamic_inference_sample(ticker, target_date, data_cfg)
    return dynamic.record, dynamic.news_count, dynamic.outcome


def build_historical_inference_payload(
    ticker: str,
    target_date: str,
    data_cfg: DictConfig,
    nlp_encoder: FinBertEncoder,
    fusion_mode: FusionMode,
) -> HistoricalInferencePayload:
    """Build Triton-ready arrays for a dynamically assembled window."""
    dynamic = build_dynamic_inference_sample(ticker, target_date, data_cfg)
    sample = dynamic.record
    if fusion_mode == "late":
        text_embedding = encode_news(nlp_encoder, sample.news_text)
        return HistoricalInferencePayload(
            time_series=sample.time_series,
            fusion_mode="late",
            news_analyzed=dynamic.news_count,
            text_embedding=text_embedding,
            outcome=dynamic.outcome,
            window_start_date=dynamic.window_start_date,
            window_end_date=dynamic.window_end_date,
        )
    text_per_step = encode_early_per_step(nlp_encoder, sample.daily_news)
    return HistoricalInferencePayload(
        time_series=sample.time_series,
        fusion_mode="early",
        news_analyzed=dynamic.news_count,
        text_per_step=text_per_step,
        outcome=dynamic.outcome,
        window_start_date=dynamic.window_start_date,
        window_end_date=dynamic.window_end_date,
    )


__all__ = [
    "HistoricalInferencePayload",
    "build_historical_inference_payload",
    "find_test_sample",
    "validate_target_date_2023",
]
