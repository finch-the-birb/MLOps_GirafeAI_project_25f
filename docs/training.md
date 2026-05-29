# Обучение

## Hybrid-модель

```bash
make train
# эквивалент:
# uv run python -m stock_forecaster.train +experiment=train
```

- Конфиг: `conf/experiment/train.yaml`
- MLflow: http://127.0.0.1:8080 (запускается автоматически)
- Чекпоинты: `checkpoints/hybrid/best-*.ckpt`

## MLflow

Логируются:

- метрики: `train_loss`, `val_loss`, `val_accuracy`, `val_f1`, `val_recall`, `val_roc_auc`
- гиперпараметры Hydra
- git commit tag
- plots → `plots/hybrid/`
- при `log_model: true`: checkpoint, PyTorch model, ONNX → `artifacts/hybrid.onnx`

!!! note "ONNX после train"
Обучение экспортирует ONNX в `artifacts/` и MLflow, но **не** копирует в Triton.
Для деплоя: `make export`.

## Baseline (tabular, OHLCV-only)

По умолчанию Random Forest; альтернативы — `hist_gradient_boosting`, `xgboost` (`conf/baseline/default.yaml`).

```bash
make baseline
```

Метрики: `artifacts/baseline_rf_metrics.json`.

## Dense-обучение (stride=1)

Для экспериментов ближе к inference-style eval — перекрывающиеся окна:

```bash
make train-dense       # hybrid → checkpoints/hybrid_dense/
make baseline-dense    # RF → artifacts/baseline_rf_dense_metrics.json
```

Утром после обучения hybrid:

```bash
make eval-test-dense eval-dense-hybrid
```

Подробнее: [Оценка → dense-прогон](evaluation.md#obuchenie-i-otsenka-na-dense-oknakh-nochnoy-progon).

## Trainer defaults

`conf/trainer/default.yaml`:

- AdamW, lr=5e-5, weight_decay=0.03
- EarlyStopping на `val_loss`, patience=5
- `use_class_weights: true` (BCE pos_weight)

## Воспроизводимость

- `seed: 42` в `conf/config.yaml`
- фиксированные сплиты по годам
