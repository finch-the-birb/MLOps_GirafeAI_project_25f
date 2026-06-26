"""FastAPI HTTP service: preprocess inputs and query Triton."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from stock_forecaster.service.routes_v1 import register_v1_routes
from stock_forecaster.service.routes_v1 import router as v1_router

logger = logging.getLogger(__name__)


def _load_cfg() -> DictConfig:
    conf_dir = Path(__file__).resolve().parents[3] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(config_name="config")


@lru_cache(maxsize=1)
def _get_nlp_encoder(cfg: DictConfig):
    encoder = instantiate(cfg.model.nlp_encoder)
    encoder.eval()
    return encoder


def create_app(cfg: DictConfig | None = None) -> FastAPI:
    app = FastAPI(title="Stock Forecaster API", version="0.1.0")
    runtime_cfg = cfg or _load_cfg()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    register_v1_routes(v1_router, runtime_cfg=runtime_cfg, get_nlp_encoder=_get_nlp_encoder)
    app.include_router(v1_router)

    return app


app = create_app()


def main() -> None:
    cfg = _load_cfg()
    uvicorn.run(
        "stock_forecaster.service.app:app",
        host=cfg.inference.service_host,
        port=cfg.inference.service_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
