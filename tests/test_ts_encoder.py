"""Smoke tests for TSLib-backed iTransformer encoder."""

import torch

from stock_forecaster.models.ts_encoder import TSLibITransformerEncoder


def test_tslib_itransformer_forward_shape() -> None:
    encoder = TSLibITransformerEncoder(
        seq_len=30,
        n_features=6,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        latent_dim=64,
    )
    batch = torch.randn(4, 30, 6)
    latent = encoder(batch)
    assert latent.shape == (4, 64)
