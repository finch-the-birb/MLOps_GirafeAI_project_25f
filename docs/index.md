# Stock Forecaster MLOps

Мультимодальная система прогнозирования **направления** цены акции на следующий торговый день
по историческим котировкам и point-in-time новостям (FNSPID).

**Автор:** Евсеев Иван Александрович

## Задача

- **Вход:** окно OHLCV `(batch, 30, 6)` + новости по дням (early fusion, FinBERT).
- **Выход:** вероятность роста; метка `1`, если доходность > **0.3%**.
- **Сплиты:** train 2018–2021 · val 2022 · test 2023 (time series split, без shuffle).

## Стек

| Компонент  | Технология                            |
| ---------- | ------------------------------------- |
| Данные     | DVC + HuggingFace FNSPID              |
| Конфиг     | Hydra `conf/`                         |
| Обучение   | PyTorch Lightning + MLflow            |
| Модель     | iTransformer + FinBERT + gated fusion |
| Production | ONNX → NVIDIA Triton                  |
| UI         | FastAPI + Streamlit                   |

## Быстрые команды

```bash
make install
make dvc-pull
make train
make export
make stack-up
```

Полный список: `make help` или раздел [Быстрый старт](getting-started.md).

## Структура репозитория

```text
MLOps/
├── conf/                  # Hydra
├── docs/                  # документация MkDocs (этот сайт)
├── src/stock_forecaster/  # основной пакет
├── src/ui/                # Streamlit
├── triton_model_repo/     # Triton config.pbtxt (+ model.onnx локально)
├── Makefile
├── mkdocs.yml
└── infer.py
```

Краткое описание для GitHub — корневой `README.md` в репозитории.
