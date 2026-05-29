# Оценка

## Test split (stride=30)

```bash
make eval
```

- Конфиг: `conf/eval/test.yaml`
- Выход: `artifacts/metrics_test2023.json`
- Тот же dataloader, что при обучении (неперекрывающиеся окна, ~456 сэмплов на test 2023).

## Dense-оценка (inference-style)

```bash
make eval-dense
```

- Конфиг: `conf/eval/dense.yaml`
- Каждый торговый день 2023, dynamic windows (как Streamlit/API).
- Выход: `artifacts/metrics_dense_test2023.json`
- Включает reference-метрики stride=30 для сравнения.

## Метрики успеха (целевые)

| Метрика   | Цель   | Зачем                     |
| --------- | ------ | ------------------------- |
| Precision | > 55%  | Меньше ложных «buy»       |
| Accuracy  | > 54%  | Общая точность            |
| F1        | > 50%  | Баланс при дисбалансе     |
| Recall    | > 50%  | Не предсказывать всегда 0 |
| ROC-AUC   | > 0.55 | Разделение классов        |

Случайное угадывание ≈ 50%; edge в production часто от 53–55%.

## Baseline vs hybrid (test 2023, stride=30)

!!! warning "Не dense-сравнение"
Таблица ниже — **test 2023**, `window_stride=30`, порог вероятности **0.5**.
Это тот же режим, что при обучении (`make train` / `make baseline`), **не** dense-оценка
(`make eval-dense`), где окна строятся как в Streamlit/API на каждый торговый день.
Сравнивать RF и hybrid между собой здесь корректно; для inference-style метрик см. раздел ниже.

Порог **0.5** для обеих моделей (hybrid — sigmoid ≥ 0.5; RF — `predict_proba` ≥ 0.5).

| Метрика   | RF baseline | Hybrid (gated) | Δ (hybrid − RF) |
| --------- | ----------- | -------------- | --------------- |
| Accuracy  | 0.537       | 0.594          | +0.057          |
| Precision | 0.426       | 0.517          | +0.091          |
| Recall    | 0.269       | 0.617          | +0.348          |
| F1        | 0.330       | 0.563          | +0.233          |
| ROC-AUC   | 0.499       | 0.636          | +0.137          |
| n (test)  | 456         | ~456           | —               |

Источники:

- **RF baseline** — `make baseline` → `artifacts/baseline_rf_metrics.json` (`test_metrics_default_0.5`).
- **Hybrid** — лучший чекпоинт `gated_fusion`, thr=0.3%, stride=30 (`make eval` / MLflow run `gated_hybrid_thr03_lb30_dm96_do30`).

RF при пороге, подобранном на val (F1-optimal ≈ 0.39), даёт F1 ≈ 0.55, но AUC остаётся ≈ 0.50 — рост F1 за счёт recall, без улучшения ранжирования.

## Dense-метрики (inference-style)

Отдельный режим — **не** смешивать с таблицей выше.

| Модель | Режим                               | F1    | AUC   | n    |
| ------ | ----------------------------------- | ----- | ----- | ---- |
| Hybrid | Dense eval (`make eval-dense`)      | ~0.41 | ~0.49 | ~14k |
| Hybrid | Stride-30 reference (в том же JSON) | ~0.56 | ~0.64 | ~472 |

Dense ближе к реальному сценарию «предсказать на любой день», поэтому метрики ниже, чем при stride=30.

## Обучение и оценка на dense-окнах (ночной прогон)

Чтобы обучить модели на **перекрывающихся окнах** (`window_stride=1`) и утром посмотреть метрики:

### 1. Запуск обучения (вечером)

```bash
make train-dense      # hybrid → checkpoints/hybrid_dense/
make baseline-dense   # RF → artifacts/baseline_rf_dense_metrics.json
```

Конфиги:

| Команда               | Experiment                            | Data                         | Артефакты                                          |
| --------------------- | ------------------------------------- | ---------------------------- | -------------------------------------------------- |
| `make train-dense`    | `conf/experiment/train_dense.yaml`    | `conf/data/thr03_dense.yaml` | `checkpoints/hybrid_dense/`, MLflow `hybrid_dense` |
| `make baseline-dense` | `conf/experiment/baseline_dense.yaml` | `conf/data/thr03_dense.yaml` | `artifacts/baseline_rf_dense_metrics.json`         |

!!! note "Время обучения"
`train-dense` заметно дольше `make train`: ~30× больше train-окон (stride 1 vs 30).
Запускайте на GPU; RF baseline обычно завершается быстрее hybrid.

### 2. Оценка hybrid_dense (утром)

После обучения подставьте путь к лучшему чекпоинту (или оставьте авто-поиск в `checkpoints/hybrid_dense/`):

```bash
# stride=1 test loader (как при train-dense)
make eval-test-dense

# inference-style dense (как Streamlit) + stride-30 reference
make eval-dense-hybrid

# переопределить чекпоинт вручную:
make eval-dense-hybrid HYBRID_DENSE_CKPT=checkpoints/hybrid_dense/best-epoch=05-val_loss=0.81.ckpt
```

Выходы:

| Команда                  | JSON                                                     |
| ------------------------ | -------------------------------------------------------- |
| `make eval-test-dense`   | `artifacts/metrics_hybrid_dense_stride1_test2023.json`   |
| `make eval-dense-hybrid` | `artifacts/metrics_hybrid_dense_inference_test2023.json` |

### 3. Метрики RF dense

Baseline пишет метрики сразу при обучении:

```bash
cat artifacts/baseline_rf_dense_metrics.json
# test_metrics_default_0.5 — accuracy, precision, recall, f1, roc_auc
```

Сравнение RF dense vs hybrid dense — по полям `test_metrics_default_0.5` (RF) и `test_metrics_dense` / `test_metrics_stride_reference` (hybrid из `make eval-dense-hybrid`).

### Полный сценарий одной строкой

```bash
make train-dense baseline-dense && make eval-test-dense eval-dense-hybrid
```

(вторую часть — после завершения `train-dense`).
