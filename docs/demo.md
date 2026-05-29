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

Triton на `/v2/health/ready` и `.../models/.../ready` возвращает **HTTP 200 с пустым телом** —
это нормально. Для скрина используйте команды, которые **печатают видимый вывод**:

```powershell
# код ответа (ожидается 200)
(Invoke-WebRequest -Uri "http://127.0.0.1:8000/v2/health/ready" -UseBasicParsing).StatusCode

# метаданные модели (JSON — удобно для скрина)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v2/models/multimodal_model_early" | ConvertTo-Json -Depth 5

# smoke-test инференса (probability=...)
make test-triton

# curl: показать код, если тело пустое
curl.exe -s -w "HTTP %{http_code}`n" http://127.0.0.1:8000/v2/health/ready
curl.exe -s http://127.0.0.1:8000/v2/models/multimodal_model_early
```

Скрин: `HTTP 200`, JSON модели или вывод `make test-triton` (`probability=...`).

### 07 — FastAPI `/api/v1/predict`

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/health"

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/api/v1/predict" `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","target_date":"2023-06-15"}'

# curl в PowerShell: нужен --%, иначе JSON «ломается» при передаче в curl.exe
curl.exe --% -s -X POST http://127.0.0.1:8001/api/v1/predict -H "Content-Type: application/json" -d "{\"ticker\":\"AAPL\",\"target_date\":\"2023-06-15\"}"
```

Скрин: JSON-ответ в терминале или Swagger → http://127.0.0.1:8001/docs .
