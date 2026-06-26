# Данные и DVC

## Источник

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) — Financial News and Stock Price Integration Dataset.

- Полный корпус ~30 ГБ.
- В проекте: топ-80 тикеров → `data/processed/fnspid_subset_thr03.parquet` (~32 МБ).
- Порог метки: **0.3%** (`label_threshold_pct`).

## Point-in-time

Новости для даты T включают только статьи с `date <= T`. Окно `seq_len=30` заканчивается днём **до** target_date.

## Сплиты

| Split | Годы      |
| ----- | --------- |
| Train | 2018–2021 |
| Val   | 2022      |
| Test  | 2023      |

Без случайного перемешивания (time series split).

## Как получить данные после `git clone`

Parquet **не в Git** (см. `.gitignore`). Публичного DVC-remote в репозитории нет — основной путь для проверяющего:

```bash
make install
make data
```

`make data` вызывает `python -m stock_forecaster.data.download_data`, читает параметры из `dvc/params.yaml` (80 тикеров, порог 0.3%) и собирает parquet из HuggingFace.

!!! warning "Время и объём"
    Первый запуск скачивает с HuggingFace ~23 ГБ news CSV и ~560 МБ zip с котировками.
    Повторный запуск использует кэш в `data/processed/raw/hf/`.
    Ориентир: **15–40 минут** в зависимости от сети.

Проверка:

```bash
# Windows PowerShell
(Get-Item data/processed/fnspid_subset_thr03.parquet).Length / 1MB
# ожидается ~31–33 МБ
```

Пересборка с нуля:

```bash
uv run python -m stock_forecaster.data.download_data --force
```

## DVC (опционально, для владельца репозитория)

```yaml
# dvc.yaml — стадия prepare, параметры в dvc/params.yaml
stages:
  prepare:
    cmd: python -m stock_forecaster.data.download_data
    params:
      - dvc/params.yaml:
          - prepare.top_tickers
          - prepare.label_threshold_pct
          - prepare.processed_file
    outs:
      - data/processed/fnspid_subset_thr03.parquet
```

| Файл | Назначение |
| ---- | ---------- |
| `dvc/params.yaml` | Параметры стадии `prepare` (тикеры, порог, путь) |
| `dvc.lock` | Хеши артефактов (в Git) |
| `.dvc/config` | Remote: `localstorage` → `../dvc-storage` (локально, не в Git) |

### Makefile

| Команда           | Когда |
| ----------------- | ----- |
| `make data`       | **После clone** — собрать parquet из HuggingFace |
| `make dvc-pull`   | Если настроен свой remote с уже выложенным parquet |
| `make dvc-push`   | После `make data` — выложить parquet в свой remote |
| `make dvc-status` | Проверить cache ↔ remote |

```bash
# сценарий для проверяющего (без своего DVC remote)
make install
make data
make train

# сценарий владельца с настроенным remote (например S3 в .dvc/config.local)
make data
dvc repro          # обновить dvc.lock при изменении dvc/params.yaml
make dvc-push
```

Чтобы раздавать данные через DVC, настройте remote в `.dvc/config.local` (пример S3 закомментирован в `.dvc/config`) и выполните `make dvc-push` после сборки.

Pull также вызывается из Python при обучении: `pull_with_dvc()` в `download_data.py` (если remote доступен).

Гиперпараметры обучения — в Hydra `conf/` (см. [Конфигурация](configuration.md)).

## Что не коммитить

`.gitignore` исключает `*.parquet`, `dvc-storage/`, `data/processed/*`.

## Пример сэмпла

```json
{
  "ticker": "AAPL",
  "target_date": "2023-01-15",
  "time_series": [{ "date": "2023-01-14", "open": 150.0, "close": 151.0, "...": "..." }],
  "daily_news": ["Apple announces new VR headset launch date."],
  "target_label": 1
}
```
