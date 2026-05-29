# Данные и DVC

## Источник

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) — Financial News and Stock Price Integration Dataset.

- Полный корпус ~30 ГБ.
- В проекте: топ-80 тикеров → `data/processed/fnspid_subset_thr03.parquet`.
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

## DVC

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python -m stock_forecaster.data.download_data
    outs:
      - data/processed/fnspid_subset_thr03.parquet
```

- Lock-файл: `dvc.lock` (в Git).
- Remote по умолчанию: `localstorage` → `dvc-storage/` (в `.gitignore`).
- Конфиг: `.dvc/config`.

### Makefile

| Команда           | Когда                                                 |
| ----------------- | ----------------------------------------------------- |
| `make dvc-pull`   | После `git clone` — подтянуть parquet из remote       |
| `make dvc-push`   | После `make data` или `dvc repro` — выложить в remote |
| `make dvc-status` | Проверить, что cache и remote синхронизированы        |
| `make data`       | Собрать parquet из HuggingFace (если файла ещё нет)   |

```bash
# типичный сценарий для проверяющего
make install
make dvc-pull

# после локальной пересборки датасета
make data
make dvc-push
make dvc-status
```

Pull также доступен из Python: `pull_with_dvc()` в `download_data.py`.

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
