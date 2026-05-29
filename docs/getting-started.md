# Быстрый старт

## Установка

```bash
make install          # uv sync + pre-commit
make install-ui       # Streamlit (опционально)
```

## Данные

```bash
make dvc-pull         # после git clone — из DVC remote
# или make data       # первая сборка из HuggingFace (долго)
make dvc-push         # после make data — выложить в remote
make dvc-status       # cache ↔ remote синхронизированы?
```

Parquet: `data/processed/fnspid_subset_thr03.parquet` (не в Git, DVC). Подробнее: [Данные и DVC](data-and-dvc.md).

## Обучение и эксперименты

```bash
make mlflow           # http://127.0.0.1:8080
make train            # hybrid; MLflow стартует автоматически
make baseline         # tabular baseline (RF по умолчанию, OHLCV-only)
make eval             # test 2023, stride=30
make eval-dense       # все торговые дни 2023
```

Чекпоинты: `checkpoints/hybrid/` (в `.gitignore`).

## Production

```bash
make export           # ONNX → artifacts/hybrid.onnx + triton_model_repo/
make triton-up
make test-triton
make stack-up         # Triton + FastAPI :8001 + Streamlit :8501
make infer TICKER=AAPL
make stop
```

## Документация

```bash
make docs-serve       # http://127.0.0.1:8002
make docs-build       # site/ (статический HTML)
```

## Полезные команды

| Команда       | Действие                |
| ------------- | ----------------------- |
| `make lint`   | pre-commit run -a       |
| `make test`   | pytest                  |
| `make verify` | smoke-test Hydra config |
