"""Clean stale data artifacts and rebuild fnspid_subset.parquet with full news (80 tickers)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Removed {path}")


def clean_data_cache() -> None:
    """Drop processed/raw HF cache and old parquet; keep checkpoints unless --all-checkpoints."""
    targets = [
        ROOT / "data" / "processed" / "raw",
        ROOT / "data" / "processed" / "fnspid_subset.parquet",
    ]
    for target in targets:
        _rm(target)


def clean_sweep_checkpoints() -> None:
    sweep_dir = ROOT / "checkpoints" / "hyperparam_sweep"
    _rm(sweep_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-tickers", type=int, default=80)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print("Cleaning old processed data...")
    clean_data_cache()
    if args.all_checkpoints:
        print("Cleaning hyperparam_sweep checkpoints...")
        clean_sweep_checkpoints()

    if args.skip_download:
        return

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "stock_forecaster.data.download_data",
        "--force",
        f"--top-tickers={args.top_tickers}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
