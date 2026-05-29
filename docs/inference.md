# Инференс

## Цепочка

```text
Streamlit / CLI ──► FastAPI ──► FinBERT (preprocess) ──► Triton (ONNX) ──► probability
```

| Сервис    | Порт | Роль                          |
| --------- | ---- | ----------------------------- |
| Triton    | 8000 | ONNX `multimodal_model_early` |
| FastAPI   | 8001 | Preprocess + HTTP API         |
| Streamlit | 8501 | UI demo (2023)                |

## Экспорт ONNX

```bash
make export
```

- Скрипт: `export_onnx.py`, конфиг `conf/export.yaml`
- Артефакт: `artifacts/hybrid.onnx`
- Копия для Triton: `triton_model_repo/multimodal_model_early/1/model.onnx`

!!! warning "ONNX не в Git"
Файл `.onnx` в `.gitignore`. После клонирования выполните `make export`
(нужен локальный checkpoint).

## Triton

`triton_model_repo/multimodal_model_early/config.pbtxt`:

- platform: `onnxruntime_onnx`
- inputs: `time_series [30,6]`, `text_per_step [30,32]`
- output: `probability [1]`

```bash
make triton-up
make test-triton
```

=== "PowerShell"

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v2/health/ready"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v2/models/multimodal_model_early/ready"
```

=== "Git Bash / Linux"

```bash
curl -s http://127.0.0.1:8000/v2/health/ready
curl -s http://127.0.0.1:8000/v2/models/multimodal_model_early/ready
```

!!! tip "Windows"
В PowerShell `curl` — alias `Invoke-WebRequest` (спрашивает подтверждение).
Используйте `Invoke-RestMethod`, `curl.exe` или `Invoke-WebRequest -UseBasicParsing`.

## FastAPI

```bash
make api
# Swagger: http://127.0.0.1:8001/docs
```

Historical predict (2023):

=== "PowerShell"

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/api/v1/predict" `
  -ContentType "application/json" `
  -Body '{"ticker":"AAPL","target_date":"2023-06-15"}'
```

=== "Git Bash / Linux"

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "target_date": "2023-06-15"}'
```

## Streamlit

```bash
make stack-up    # Triton + API + UI
# http://127.0.0.1:8501
```

## CLI

```bash
make infer TICKER=AAPL
# или: uv run python infer.py ticker=AAPL
```

## TensorRT (опционально)

```bash
bash scripts/compile_tensorrt.sh artifacts/hybrid.onnx artifacts/hybrid.engine
```

Shapes для early fusion: `text_per_step` вместо legacy `text_embedding [768]`.
