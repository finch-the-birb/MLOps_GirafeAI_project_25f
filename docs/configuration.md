# Конфигурация (Hydra)

Корень: `conf/config.yaml` — defaults `data=thr03`, `model=hybrid`.

## Структура

| Файл                                  | Назначение                                    |
| ------------------------------------- | --------------------------------------------- |
| `conf/data/default.yaml`              | Базовая схема FNSPID                          |
| `conf/data/thr03.yaml`                | Финальный датасет (thr=0.3%)                  |
| `conf/model/early.yaml`               | Early fusion base                             |
| `conf/model/hybrid.yaml`              | Gated fusion                                  |
| `conf/trainer/default.yaml`           | AdamW, early stopping                         |
| `conf/mlflow/default.yaml`            | URI :8080                                     |
| `conf/inference/default.yaml`         | Triton, порты                                 |
| `conf/experiment/train.yaml`          | Рецепт обучения (stride=30)                   |
| `conf/experiment/train_dense.yaml`    | Hybrid на dense-окнах (stride=1)              |
| `conf/experiment/baseline.yaml`       | Tabular baseline (RF по умолчанию, stride=30) |
| `conf/experiment/baseline_dense.yaml` | Tabular baseline dense (stride=1)             |
| `conf/data/thr03_dense.yaml`          | Датасет thr=0.3%, window_stride=1             |
| `conf/eval/test.yaml`                 | Eval test (stride=30)                         |
| `conf/eval/dense.yaml`                | Dense eval (inference-style)                  |
| `conf/eval/test_hybrid_dense.yaml`    | Test eval hybrid_dense                        |
| `conf/eval/dense_hybrid_dense.yaml`   | Dense eval hybrid_dense                       |
| `conf/export.yaml`                    | ONNX export                                   |

## Примеры override

```bash
uv run python -m stock_forecaster.train +experiment=train trainer.max_epochs=5
uv run python -m stock_forecaster.evaluate --config-name eval/test \
  checkpoint_path=checkpoints/hybrid/best-epoch=02-val_loss=0.8134.ckpt
```

## Instantiation

Модели создаются через `_target_`:

```yaml
# conf/model/early.yaml
_target_: stock_forecaster.models.early_fusion_model.EarlyFusionStockModel
```

Tabular baseline: `baseline.model_type` — `random_forest` (default), `hist_gradient_boosting`, `xgboost`.
