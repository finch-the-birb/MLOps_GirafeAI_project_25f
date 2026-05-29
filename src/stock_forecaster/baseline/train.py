"""Train OHLCV-only tabular baseline (Random Forest / boosting) — proposal §6."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from stock_forecaster.baseline.features import dataset_to_xy
from stock_forecaster.baseline.fil_backend import load_fil_predict_proba, save_sklearn_for_fil
from stock_forecaster.baseline.metrics import best_threshold_f1, classification_metrics
from stock_forecaster.data.datamodule import FnspidDataModule
from stock_forecaster.data.download_data import download_data
from stock_forecaster.utils.git_info import get_git_commit_id

logger = logging.getLogger(__name__)


def _log_mlflow_run(
    cfg: DictConfig,
    model: Any,
    payload: dict[str, Any],
    native_path: Path,
    metrics_path: Path,
    fil_path: Path | None,
) -> str | None:
    """Log tabular baseline to the same MLflow experiment as PyTorch training."""
    if not cfg.get("mlflow"):
        return None

    # Avoid UnicodeEncodeError on Windows consoles when MLflow prints run URLs.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1")

    tracking_uri = str(cfg.mlflow.tracking_uri)
    experiment_name = str(cfg.mlflow.experiment_name)
    run_name = str(cfg.baseline.run_name)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", get_git_commit_id())
        mlflow.set_tag("model_family", "tabular_baseline")
        mlflow.set_tag("modalities", "ohlcv_only")

        mlflow.log_params(
            {
                "baseline.model_type": str(cfg.baseline.model_type),
                "data.seq_len": int(cfg.data.seq_len),
                "data.window_stride": int(cfg.data.get("window_stride", 1)),
                "n_tabular_features": int(payload["n_tabular_features"]),
            }
        )
        params = OmegaConf.to_container(cfg.baseline.get("params", {}), resolve=True)
        if isinstance(params, dict):
            for key, value in params.items():
                mlflow.log_param(f"baseline.params.{key}", value)

        for split_name, metrics in (
            ("val", payload.get("val_metrics_at_best_threshold", {})),
            ("test_default", payload.get("test_metrics_default_0.5", {})),
            ("test_tuned", payload.get("test_metrics_tuned_threshold", {})),
        ):
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if metric_name == "threshold":
                    mlflow.log_param(f"{split_name}_threshold", float(value))
                else:
                    mlflow.log_metric(f"{split_name}_{metric_name}", float(value))

        mlflow.log_param("val_best_threshold", float(payload["val_best_threshold"]))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(native_path), artifact_path="model")

        if fil_path is not None and fil_path.is_file():
            mlflow.log_artifact(str(fil_path), artifact_path="fil_export")

        if cfg.mlflow.get("log_model", False):
            model_type = str(cfg.baseline.model_type)
            if model_type in ("random_forest", "hist_gradient_boosting"):
                from mlflow import sklearn as mlflow_sklearn

                mlflow_sklearn.log_model(model, artifact_path="sklearn_model")
            elif model_type == "xgboost":
                from mlflow import xgboost as mlflow_xgboost

                mlflow_xgboost.log_model(model, artifact_path="xgboost_model")

        return run.info.run_id


def _build_estimator(cfg: DictConfig) -> Any:
    model_type = str(cfg.baseline.model_type)
    params = OmegaConf.to_container(cfg.baseline.get("params", {}), resolve=True)
    if not isinstance(params, dict):
        params = {}
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    if model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            msg = "baseline.model_type=xgboost requires optional dependency xgboost"
            raise ImportError(msg) from exc
        return xgb.XGBClassifier(**params)
    msg = f"Unknown baseline.model_type: {model_type}"
    raise ValueError(msg)


def _predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return np.asarray(proba)[:, 1]
    scores = model.decision_function(x)
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scores))


def _save_native_model(model: Any, model_dir: Path, model_type: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_type == "xgboost":
        path = model_dir / "model.ubj"
        model.save_model(str(path))
        return path
    path = model_dir / "model.joblib"
    joblib.dump(model, path)
    return path


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    processed_path = Path(cfg.data.processed_file)
    download_data(processed_path, top_tickers=cfg.data.top_tickers)

    data_module = FnspidDataModule(cfg.data)
    data_module.setup("fit")

    train_ds = data_module.train_dataset
    val_ds = data_module.val_dataset
    test_ds = data_module.test_dataset
    if train_ds is None or val_ds is None or test_ds is None:
        msg = "Datasets not initialized"
        raise RuntimeError(msg)

    x_train, y_train = dataset_to_xy(train_ds)
    x_val, y_val = dataset_to_xy(val_ds)
    x_test, y_test = dataset_to_xy(test_ds)
    logger.info(
        "Tabular baseline | train=%s val=%s test=%s | features=%s",
        x_train.shape,
        x_val.shape,
        x_test.shape,
        x_train.shape[1],
    )

    model_type = str(cfg.baseline.model_type)
    model = _build_estimator(cfg)
    model.fit(x_train, y_train)

    model_dir = Path(cfg.baseline.output_dir) / str(cfg.baseline.run_name)
    native_path = _save_native_model(model, model_dir, model_type)
    fil_path = save_sklearn_for_fil(model, model_dir / "fil_export")

    val_prob = _predict_proba(model, x_val)
    test_prob_sklearn = _predict_proba(model, x_test)

    test_prob_fil: np.ndarray | None = None
    if fil_path is not None:
        test_prob_fil = load_fil_predict_proba(fil_path, x_test)

    thr, val_at_thr = best_threshold_f1(y_val, val_prob)
    test_default = classification_metrics(y_test, test_prob_sklearn, threshold=0.5)
    test_tuned = classification_metrics(y_test, test_prob_sklearn, threshold=thr)

    payload: dict[str, Any] = {
        "run_name": str(cfg.baseline.run_name),
        "model_type": model_type,
        "description": "OHLCV-only tabular baseline (no FinBERT / no fusion)",
        "train_years": data_module.train_years,
        "val_years": data_module.val_years,
        "test_years": data_module.test_years,
        "data_seq_len": int(cfg.data.seq_len),
        "data_window_stride": int(cfg.data.get("window_stride", 1)),
        "n_tabular_features": int(x_train.shape[1]),
        "model_path": str(native_path),
        "fil_export": str(fil_path) if fil_path else None,
        "val_best_threshold": thr,
        "val_metrics_at_best_threshold": val_at_thr,
        "test_metrics_default_0.5": test_default,
        "test_metrics_tuned_threshold": {**test_tuned, "threshold": thr},
    }
    if test_prob_fil is not None:
        payload["test_metrics_fil_tuned"] = {
            **classification_metrics(y_test, test_prob_fil, threshold=thr),
            "threshold": thr,
            "backend": "cuml.fil",
        }

    out_json = Path(cfg.baseline.metrics_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run_id = _log_mlflow_run(cfg, model, payload, native_path, out_json, fil_path)
    if run_id:
        payload["mlflow_run_id"] = run_id
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("MLflow run_id=%s (%s)", run_id, cfg.mlflow.tracking_uri)
    else:
        logger.info("MLflow logging skipped (no mlflow config)")

    logger.info("Wrote %s", out_json)
    logger.info("Test @0.5: %s", test_default)
    logger.info("Test @thr=%.2f: %s", thr, test_tuned)


if __name__ == "__main__":
    main()
