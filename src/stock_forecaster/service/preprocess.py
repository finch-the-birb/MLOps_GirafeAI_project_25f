"""FinBERT encoding helpers for Triton inference payloads."""

from __future__ import annotations

import numpy as np
import torch

from stock_forecaster.models.nlp_encoder import FinBertEncoder


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
