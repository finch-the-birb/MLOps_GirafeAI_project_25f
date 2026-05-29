"""Optional RAPIDS FIL inference for tree models (sklearn / XGBoost exports)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def save_sklearn_for_fil(model: Any, export_dir: Path) -> Path | None:
    """
    Export a fitted sklearn tree ensemble to Treelite, for FIL.load().

    Returns path to the exported model file, or None if Treelite is unavailable.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "treelite_model.bin"
    try:
        import treelite.sklearn as tl_sklearn  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("treelite not installed; skip FIL export (pip install treelite)")
        return None

    tl_model = tl_sklearn.import_model(model)
    tl_model.export_model(str(out_path))
    return out_path


def load_fil_predict_proba(model_path: Path, x: np.ndarray) -> np.ndarray | None:
    """Run FIL predict_proba if cuml.fil is available; otherwise return None."""
    try:
        from cuml.fil import ForestInference  # type: ignore[import-untyped]
    except ImportError:
        logger.info("cuml.fil not available; use sklearn predict_proba on CPU")
        return None

    fil_model = ForestInference.load(str(model_path), output_class=True)
    batch = max(1, min(4096, len(x)))
    try:
        fil_model.optimize(batch_size=batch)
    except Exception:
        logger.debug("FIL optimize skipped", exc_info=True)

    probs = fil_model.predict_proba(x.astype(np.float32))
    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1].astype(np.float64)
    return probs.reshape(-1).astype(np.float64)
