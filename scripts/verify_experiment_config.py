"""Smoke-test Hydra config: resolve hyperparameters and run one forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"


def main(experiment: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])

    ts = cfg.model.ts_encoder
    print("=== Resolved config ===")
    print(f"fusion_mode:     {cfg.data.fusion_mode}")
    print(f"lookback seq_len: {cfg.data.seq_len}")
    print(f"d_model:         {ts.d_model}")
    print(f"e_layers:        {ts.e_layers}")
    print(f"d_ff:            {ts.d_ff}")
    print(f"n_heads:         {ts.n_heads}")
    print(f"dropout (TS):    {ts.dropout}")
    print(f"n_features (iT): {ts.n_features}")
    print(f"text_feat_dim:   {cfg.model.text_feat_dim}")
    cid = cfg.model.get("classifier_input_dim", cfg.model.get("fusion_dim"))
    print(f"classifier_input_dim: {cid}")

    model = instantiate(cfg.model)
    model.eval()
    batch = 2
    seq_len = int(cfg.data.seq_len)
    n_price = int(cfg.data.n_features)

    time_series = torch.randn(batch, seq_len, n_price)
    daily_news = [[f"headline day {d}" for d in range(seq_len)] for _ in range(batch)]

    with torch.no_grad():
        latent = model.encode(time_series, daily_news)

    expected_dim = int(cfg.model.classifier_input_dim)
    if latent.shape != (batch, expected_dim):
        msg = f"Expected latent {(batch, expected_dim)}, got {tuple(latent.shape)}"
        raise SystemExit(msg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"forward OK: latent {tuple(latent.shape)}")
    print(f"params: trainable={trainable:,} total={total:,}")
    print("OK")


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(exp)
