# Демо и скриншоты

Сценарий для защиты / GitHub README.

## Подготовка

```bash
make install
make dvc-pull         # или make data + make dvc-push при первой сборке
make export
```

## Блок A — обучение

```bash
make train
make baseline
make eval
make eval-dense
```

## Блок B — инференс

```bash
make stack-up
make infer TICKER=AAPL
make stop
```

## Скриншоты 5–7: команды (PowerShell, без запросов)

В Windows **не используйте** `curl` без `.exe` — это alias `Invoke-WebRequest` с интерактивным предупреждением.
Ниже команды работают сразу, без `[Y/N]`.

Перед скринами 6–7:

```powershell
make stack-up   # Triton + FastAPI + Streamlit
make export     # если ещё нет model.onnx в triton_model_repo
```

### 05 — RF vs hybrid

```powershell
make baseline
make eval
# Сравните artifacts/baseline_rf_metrics.json (test_metrics_default_0.5)
# и artifacts/metrics_test2023.json — таблицу/график соберите в Excel или снимите MLflow
```

### 06 — Triton READY

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v2/health/ready"

Invoke-RestMethod -Uri "http://127.0.0.1:8000/v2/models/multimodal_model_early/ready"

# альтернатива (настоящий curl, не alias):
curl.exe -s http://127.0.0.1:8000/v2/health/ready
curl.exe -s http://127.0.0.1:8000/v2/models/multimodal_model_early/ready
```

Скрин: JSON с `"ready": true` или вывод `make test-triton`.

### 07 — FastAPI `/api/v1/predict`

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/health"

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/api/v1/predict" `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","target_date":"2023-06-15"}'

# альтернатива:
curl.exe -s -X POST "http://127.0.0.1:8001/api/v1/predict" `
  -H "Content-Type: application/json" `
  -d "{\"ticker\":\"AAPL\",\"target_date\":\"2023-06-15\"}"
```

Скрин: JSON-ответ в терминале или Swagger → http://127.0.0.1:8001/docs .

## Скриншоты

Положите PNG в `docs/assets/screenshots/`:

| Файл                          | Содержание            |
| ----------------------------- | --------------------- |
| `01_mlflow_experiments.png`   | MLflow: список run'ов |
| `02_mlflow_run_metrics.png`   | Метрики hybrid run    |
| `03_mlflow_plots.png`         | Кривые обучения       |
| `04_eval_test_metrics.png`    | `make eval`           |
| `05_baseline_vs_hybrid.png`   | RF vs hybrid          |
| `06_triton_ready.png`         | Triton READY          |
| `07_fastapi_predict.png`      | `/api/v1/predict`     |
| `08_streamlit_overview.png`   | UI: тикер + календарь |
| `09_streamlit_prediction.png` | Прогноз vs факт       |
| `10_streamlit_chart.png`      | OHLCV chart           |
| `11_dense_eval.png`           | Dense eval            |

В README можно вставить:

```markdown
![MLflow](docs/assets/screenshots/01_mlflow_experiments.png)
```

Опционально: `12_docker_compose.png`, `13_cli_infer.png`.
