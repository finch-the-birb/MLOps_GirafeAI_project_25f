"""Tests for multimodal iTransformer embedding (four fusion ablation modes)."""

from types import SimpleNamespace

import pytest
import torch

from stock_forecaster.models.ts_encoder import TSLibITransformerEncoder
from stock_forecaster.third_party.tslib.layers.multimodal_embed import (
    MULTIMODAL_MODES,
    MultimodalITransformerEmbedding,
)
from stock_forecaster.third_party.tslib.models.itransformer import Model as TSLibITransformerModel


@pytest.mark.parametrize("mode", sorted(MULTIMODAL_MODES))
def test_multimodal_embedding_token_shapes(mode: str) -> None:
    batch, lookback, n_num, text_dim, d_model = 2, 96, 6, 32, 64
    embed = MultimodalITransformerEmbedding(
        seq_len=lookback,
        d_model=d_model,
        n_numeric_features=n_num,
        text_feat_dim=text_dim,
        multimodal_mode=mode,
        n_heads=4,
    )
    x_num = torch.randn(batch, lookback, n_num)
    x_text = torch.randn(batch, lookback, text_dim)
    out = embed(x_num, x_text)

    expected_tokens = 7 if mode == "flatten" else 6
    assert out.shape == (batch, expected_tokens, d_model)
    assert embed.num_tokens == expected_tokens


def test_itransformer_encode_multimodal_slices_to_six_numeric_tokens() -> None:
    """Flatten produces 7 tokens; encoder output must be sliced to 6 for the head."""
    batch, lookback, n_num, text_dim, d_model = 2, 96, 6, 32, 128
    configs = SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=lookback,
        pred_len=1,
        enc_in=n_num,
        d_model=d_model,
        n_heads=4,
        e_layers=1,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        embed="fixed",
        freq="h",
        factor=1,
        num_class=1,
        multimodal_mode="flatten",
        text_m_dim=1,
        n_numeric_features=n_num,
        text_feat_dim=text_dim,
    )
    model = TSLibITransformerModel(configs)
    x_num = torch.randn(batch, lookback, n_num)
    x_text = torch.randn(batch, lookback, text_dim)
    enc_out = model.encode_multimodal(x_num, x_text)
    assert enc_out.shape == (batch, n_num, d_model)


@pytest.mark.parametrize("mode", ["daily_mlp", "gated_fusion", "transformer_join"])
def test_ts_encoder_multimodal_latent_shape(mode: str) -> None:
    encoder = TSLibITransformerEncoder(
        seq_len=96,
        n_features=6,
        n_price_features=6,
        text_feat_dim=32,
        multimodal_mode=mode,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        latent_dim=64,
    )
    batch = 3
    x_num = torch.randn(batch, 96, 6)
    x_text = torch.randn(batch, 96, 32)
    latent = encoder(x_num, text_feats=x_text)
    assert latent.shape == (batch, 64)


def test_ts_encoder_multimodal_flatten_latent_shape() -> None:
    encoder = TSLibITransformerEncoder(
        seq_len=30,
        n_features=6,
        n_price_features=6,
        text_feat_dim=32,
        multimodal_mode="flatten",
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        latent_dim=64,
    )
    batch = 3
    x_num = torch.randn(batch, 30, 6)
    x_text = torch.randn(batch, 30, 32)
    latent = encoder(x_num, text_feats=x_text)
    assert latent.shape == (batch, 64)


def test_invalid_multimodal_mode_raises() -> None:
    with pytest.raises(ValueError, match="multimodal_mode"):
        MultimodalITransformerEmbedding(
            seq_len=96,
            d_model=64,
            multimodal_mode="pre_project",
        )
