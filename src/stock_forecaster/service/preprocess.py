"""Fetch market data and build model-ready tensors."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import torch
import yfinance as yf
from omegaconf import DictConfig

from stock_forecaster.models.nlp_encoder import FinBertEncoder


def fetch_price_window(
    ticker: str,
    seq_len: int,
    feature_columns: list[str],
) -> np.ndarray:
    """Download recent OHLCV window via Yahoo Finance."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=seq_len * 3)
    history = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date),
        progress=False,
        auto_adjust=False,
    )
    if history.empty:
        msg = f"No price data returned for ticker {ticker}"
        raise ValueError(msg)

    history = history.reset_index()
    history.columns = [str(col).lower().replace(" ", "_") for col in history.columns]
    history["change_pct"] = history["close"].pct_change() * 100.0
    history = history.dropna().tail(seq_len)

    if len(history) < seq_len:
        msg = f"Insufficient history for {ticker}: got {len(history)}, need {seq_len}"
        raise ValueError(msg)

    renamed = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "change_pct": "change_pct",
    }
    selected = history[[renamed[col] for col in feature_columns]]
    return selected.to_numpy(dtype=np.float32)


def fetch_news_placeholder(ticker: str) -> str:
    """Placeholder news aggregator; replace with licensed feed in production."""
    return f"No live news API configured. Neutral context for {ticker}."


def encode_news(
    nlp_encoder: FinBertEncoder,
    news_text: str,
) -> np.ndarray:
    """Return raw FinBERT hidden size vector before projection."""
    encoded = nlp_encoder.tokenizer(
        [news_text],
        padding=True,
        truncation=True,
        max_length=nlp_encoder.max_length,
        return_tensors="pt",
    )
    device = next(nlp_encoder.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = nlp_encoder.backbone(**encoded)
    cls_hidden = outputs.last_hidden_state[:, 0, :]
    return cls_hidden.cpu().numpy().astype(np.float32)


def encode_early_per_step(
    nlp_encoder: FinBertEncoder,
    daily_news: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Return projected per-timestep text features for early-fusion Triton/ONNX."""
    nlp_encoder.eval()
    daily_batch = [list(daily_news)]
    with torch.no_grad():
        projected = nlp_encoder.encode_sequence_batch(daily_batch)
    return projected.cpu().numpy().astype(np.float32)


def build_inference_payload(
    ticker: str,
    data_cfg: DictConfig,
    nlp_encoder: FinBertEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (time_series, text_embedding) arrays for Triton."""
    time_series = fetch_price_window(
        ticker=ticker,
        seq_len=data_cfg.seq_len,
        feature_columns=list(data_cfg.feature_columns),
    )
    news_text = fetch_news_placeholder(ticker)
    text_embedding = encode_news(nlp_encoder, news_text)
    return time_series, text_embedding
