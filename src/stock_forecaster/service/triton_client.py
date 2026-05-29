"""Thin wrapper around Triton HTTP client."""

from __future__ import annotations

import numpy as np
import tritonclient.http as httpclient

from stock_forecaster.service.fusion import FusionMode


def predict_triton(
    *,
    triton_url: str,
    model_name: str,
    model_version: str,
    time_series: np.ndarray,
    fusion_mode: FusionMode = "late",
    text_embedding: np.ndarray | None = None,
    text_per_step: np.ndarray | None = None,
) -> float:
    """Send multimodal inputs to Triton and return up probability."""
    if fusion_mode == "late":
        if text_embedding is None:
            msg = "text_embedding is required for late fusion"
            raise ValueError(msg)
        return _predict_late(
            triton_url=triton_url,
            model_name=model_name,
            model_version=model_version,
            time_series=time_series,
            text_embedding=text_embedding,
        )
    if text_per_step is None:
        msg = "text_per_step is required for early fusion"
        raise ValueError(msg)
    return _predict_early(
        triton_url=triton_url,
        model_name=model_name,
        model_version=model_version,
        time_series=time_series,
        text_per_step=text_per_step,
    )


def _predict_late(
    *,
    triton_url: str,
    model_name: str,
    model_version: str,
    time_series: np.ndarray,
    text_embedding: np.ndarray,
) -> float:
    client = httpclient.InferenceServerClient(url=triton_url, verbose=False)

    time_series_input = time_series[np.newaxis, ...].astype(np.float32)
    text_input = text_embedding.astype(np.float32)
    if text_input.ndim == 1:
        text_input = text_input[np.newaxis, ...]

    inputs = [
        httpclient.InferInput("time_series", time_series_input.shape, "FP32"),
        httpclient.InferInput("text_embedding", text_input.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(time_series_input)
    inputs[1].set_data_from_numpy(text_input)

    outputs = [httpclient.InferRequestedOutput("probability")]
    response = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=inputs,
        outputs=outputs,
    )
    probability = response.as_numpy("probability")
    return float(probability.reshape(-1)[0])


def _predict_early(
    *,
    triton_url: str,
    model_name: str,
    model_version: str,
    time_series: np.ndarray,
    text_per_step: np.ndarray,
) -> float:
    client = httpclient.InferenceServerClient(url=triton_url, verbose=False)

    time_series_input = time_series[np.newaxis, ...].astype(np.float32)
    text_input = text_per_step.astype(np.float32)
    if text_input.ndim == 2:
        text_input = text_input[np.newaxis, ...]

    inputs = [
        httpclient.InferInput("time_series", time_series_input.shape, "FP32"),
        httpclient.InferInput("text_per_step", text_input.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(time_series_input)
    inputs[1].set_data_from_numpy(text_input)

    outputs = [httpclient.InferRequestedOutput("probability")]
    response = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=inputs,
        outputs=outputs,
    )
    probability = response.as_numpy("probability")
    return float(probability.reshape(-1)[0])
