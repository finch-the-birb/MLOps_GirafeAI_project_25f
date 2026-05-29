"""Tests for fusion-mode resolution and Triton client input wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from stock_forecaster.service.fusion import (
    resolve_fusion_mode,
    triton_model_name,
    validate_model_fusion_compatibility,
)
from stock_forecaster.service.triton_client import predict_triton


def test_resolve_fusion_mode_priority() -> None:
    data_cfg = OmegaConf.create({"fusion_mode": "late"})
    inference_cfg = OmegaConf.create({"fusion_mode": "early"})
    assert resolve_fusion_mode("early", data_cfg, inference_cfg) == "early"
    assert resolve_fusion_mode(None, data_cfg, inference_cfg) == "early"
    assert resolve_fusion_mode(None, data_cfg, None) == "late"


def test_validate_model_fusion_compatibility() -> None:
    late_model = OmegaConf.create(
        {"_target_": "stock_forecaster.models.multimodal_model.MultimodalStockModel"}
    )
    early_model = OmegaConf.create(
        {"_target_": "stock_forecaster.models.early_fusion_model.EarlyFusionStockModel"}
    )
    validate_model_fusion_compatibility("late", late_model)
    validate_model_fusion_compatibility("early", early_model)
    with pytest.raises(ValueError, match="Early fusion was requested"):
        validate_model_fusion_compatibility("early", late_model)


def test_triton_model_name_early_override() -> None:
    inference_cfg = OmegaConf.create(
        {
            "triton_model_name": "multimodal_model",
            "triton_model_name_early": "multimodal_model_early",
        }
    )
    assert triton_model_name(inference_cfg, "late") == "multimodal_model"
    assert triton_model_name(inference_cfg, "early") == "multimodal_model_early"


def test_predict_triton_early_builds_text_per_step_input() -> None:
    time_series = np.zeros((30, 6), dtype=np.float32)
    text_per_step = np.zeros((1, 30, 32), dtype=np.float32)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.as_numpy.return_value = np.array([[0.42]], dtype=np.float32)
    mock_client.infer.return_value = mock_response

    with patch(
        "stock_forecaster.service.triton_client.httpclient.InferenceServerClient",
        return_value=mock_client,
    ):
        prob = predict_triton(
            triton_url="localhost:8000",
            model_name="multimodal_model_early",
            model_version="1",
            time_series=time_series,
            fusion_mode="early",
            text_per_step=text_per_step,
        )

    assert prob == pytest.approx(0.42)
    infer_args = mock_client.infer.call_args.kwargs
    input_names = [item.name() for item in infer_args["inputs"]]
    assert "text_per_step" in input_names
