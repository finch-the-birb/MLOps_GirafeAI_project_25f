.DEFAULT_GOAL := help

# Prefer Git Bash sh on Windows; fall back to default shell elsewhere.
ifeq ($(OS),Windows_NT)
SHELL := C:/Program Files/Git/bin/sh.exe
endif

UV       := uv run
PY       := uv run python
RUN_DIR  := .run
MLFLOW_PORT := 8080
API_PORT := 8001
UI_PORT  := 8501
DOCS_PORT := 8002
TICKER   ?= AAPL
TARGET_DATE ?= 2023-06-15
HYBRID_DENSE_CKPT ?= $(shell ls -t checkpoints/hybrid_dense/*.ckpt 2>/dev/null | head -1)

.PHONY: help install install-ui install-docs data dvc-pull dvc-push dvc-status lint test verify \
	mlflow mlflow-stop \
	train train-dense baseline baseline-dense \
	eval eval-dense eval-test-dense eval-dense-hybrid \
	export triton-up triton-down test-triton \
	api api-stop ui ui-stop \
	infer stack-up stack-down stop \
	docs docs-stop docs-serve docs-build

help: ## Показать команды
	@echo "MLOps Stock Forecaster — основные команды:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-18s %s\n", $$1, $$2}'

install: ## Установить зависимости (uv sync + pre-commit)
	uv sync --all-extras
	uv run pre-commit install

install-ui: ## Установить UI-зависимости (Streamlit, Plotly)
	uv pip install ".[ui]"

install-docs: ## Установить MkDocs и тему Material
	uv pip install "mkdocs>=1.6" "mkdocs-material>=9.5" "pymdown-extensions>=10.8"

data: ## Собрать данные (HuggingFace → parquet, если нет локально)
	$(PY) -m stock_forecaster.data.download_data

dvc-pull: ## Скачать DVC-артефакты из remote (после git clone)
	$(UV) dvc pull

dvc-push: ## Загрузить DVC-артефакты в remote (после make data / dvc repro)
	$(UV) dvc push

dvc-status: ## Статус DVC: локальный cache vs remote
	$(UV) dvc status --cloud

lint: ## Проверка кода (pre-commit run -a)
	uv run pre-commit run -a

test: ## Запустить pytest
	uv run pytest tests/ -q

verify: ## Smoke-test конфига Hydra (+experiment=train)
	$(PY) scripts/verify_experiment_config.py train

mlflow: $(RUN_DIR) ## Запустить MLflow UI (фон, :8080)
	@if [ -f $(RUN_DIR)/mlflow.pid ] && kill -0 $$(cat $(RUN_DIR)/mlflow.pid) 2>/dev/null; then \
		echo "MLflow уже запущен (pid $$(cat $(RUN_DIR)/mlflow.pid), http://127.0.0.1:$(MLFLOW_PORT))"; \
	else \
		echo "Запуск MLflow на http://127.0.0.1:$(MLFLOW_PORT) ..."; \
		nohup $(UV) mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) \
			> $(RUN_DIR)/mlflow.log 2>&1 & \
		echo $$! > $(RUN_DIR)/mlflow.pid; \
		sleep 2; \
		echo "MLflow pid $$(cat $(RUN_DIR)/mlflow.pid), лог: $(RUN_DIR)/mlflow.log"; \
	fi

mlflow-stop: ## Остановить MLflow
	$(call STOP_SERVICE,mlflow,$(MLFLOW_PORT))

train: mlflow ## Обучить hybrid-модель (+experiment=train, stride=30)
	$(PY) -m stock_forecaster.train +experiment=train

train-dense: mlflow ## Обучить hybrid на dense-окнах (+experiment=train_dense, stride=1)
	$(PY) -m stock_forecaster.train +experiment=train_dense

baseline: mlflow ## Tabular baseline, RF по умолчанию (OHLCV-only, stride=30)
	$(PY) -m stock_forecaster.baseline.train +experiment=baseline

baseline-dense: mlflow ## Tabular baseline на dense-окнах (stride=1)
	$(PY) -m stock_forecaster.baseline.train +experiment=baseline_dense

eval: ## Оценка hybrid на test split (stride=30)
	$(PY) -m stock_forecaster.evaluate --config-name eval/test

eval-dense: ## Dense-оценка hybrid (inference-style, все торговые дни 2023)
	$(PY) -m stock_forecaster.evaluate_dense --config-name eval/dense

eval-test-dense: ## Test eval hybrid_dense (stride=1 loader)
	@if [ -z "$(HYBRID_DENSE_CKPT)" ]; then \
		echo "Чекпоинт не найден: checkpoints/hybrid_dense/*.ckpt (сначала make train-dense)"; \
		exit 1; \
	fi
	$(PY) -m stock_forecaster.evaluate --config-name eval/test_hybrid_dense \
		checkpoint_path=$(HYBRID_DENSE_CKPT)

eval-dense-hybrid: ## Dense inference eval для hybrid_dense
	@if [ -z "$(HYBRID_DENSE_CKPT)" ]; then \
		echo "Чекпоинт не найден: checkpoints/hybrid_dense/*.ckpt (сначала make train-dense)"; \
		exit 1; \
	fi
	$(PY) -m stock_forecaster.evaluate_dense --config-name eval/dense_hybrid_dense \
		checkpoint_path=$(HYBRID_DENSE_CKPT)

export: ## Экспорт ONNX для Triton
	$(PY) -m stock_forecaster.export_onnx --config-name export

triton-up: ## Поднять Triton (docker compose)
	docker compose up -d

triton-down: ## Остановить Triton
	docker compose down

test-triton: ## Smoke-test Triton client
	$(PY) scripts/test_triton_inference.py

api: $(RUN_DIR) ## Запустить FastAPI (фон, :8001)
	@if [ -f $(RUN_DIR)/api.pid ] && kill -0 $$(cat $(RUN_DIR)/api.pid) 2>/dev/null; then \
		echo "FastAPI уже запущен (pid $$(cat $(RUN_DIR)/api.pid), http://127.0.0.1:$(API_PORT))"; \
	else \
		echo "Запуск FastAPI на http://127.0.0.1:$(API_PORT) ..."; \
		nohup $(UV) uvicorn stock_forecaster.service.app:app --host 127.0.0.1 --port $(API_PORT) \
			> $(RUN_DIR)/api.log 2>&1 & \
		echo $$! > $(RUN_DIR)/api.pid; \
		sleep 2; \
		echo "FastAPI pid $$(cat $(RUN_DIR)/api.pid)"; \
	fi

api-stop: ## Остановить FastAPI
	$(call STOP_SERVICE,api,$(API_PORT))

ui: install-ui $(RUN_DIR) ## Запустить Streamlit UI (фон, :8501)
	@if [ -f $(RUN_DIR)/ui.pid ] && kill -0 $$(cat $(RUN_DIR)/ui.pid) 2>/dev/null; then \
		echo "Streamlit уже запущен (pid $$(cat $(RUN_DIR)/ui.pid), http://127.0.0.1:$(UI_PORT))"; \
	else \
		echo "Запуск Streamlit на http://127.0.0.1:$(UI_PORT) ..."; \
		nohup $(UV) streamlit run src/ui/app.py --server.headless true --server.port $(UI_PORT) \
			> $(RUN_DIR)/ui.log 2>&1 & \
		echo $$! > $(RUN_DIR)/ui.pid; \
		sleep 2; \
		echo "Streamlit pid $$(cat $(RUN_DIR)/ui.pid)"; \
	fi

ui-stop: ## Остановить Streamlit
	$(call STOP_SERVICE,ui,$(UI_PORT))

infer: triton-up ## CLI-инференс (TICKER=AAPL, TARGET_DATE=2023-06-15)
	$(PY) infer.py inference.ticker=$(TICKER) inference.target_date=$(TARGET_DATE)

stack-up: triton-up api ui ## Поднять стек инференса (Triton + FastAPI + Streamlit)
	@echo "Стек готов: Triton :8000, API :$(API_PORT), UI :$(UI_PORT)"

stack-down: api-stop ui-stop triton-down ## Остановить стек инференса
	@echo "Стек инференса остановлен."

stop: mlflow-stop stack-down docs-stop ## Остановить все фоновые сервисы
	@echo "Все сервисы остановлены."

docs: install-docs $(RUN_DIR) ## Запустить MkDocs (фон, :8002)
	@if [ -f $(RUN_DIR)/docs.pid ] && kill -0 $$(cat $(RUN_DIR)/docs.pid) 2>/dev/null; then \
		echo "MkDocs уже запущен (pid $$(cat $(RUN_DIR)/docs.pid), http://127.0.0.1:$(DOCS_PORT))"; \
	else \
		echo "Запуск MkDocs на http://127.0.0.1:$(DOCS_PORT) ..."; \
		nohup $(UV) mkdocs serve -a 127.0.0.1:$(DOCS_PORT) \
			> $(RUN_DIR)/docs.log 2>&1 & \
		echo $$! > $(RUN_DIR)/docs.pid; \
		sleep 2; \
		echo "MkDocs pid $$(cat $(RUN_DIR)/docs.pid), лог: $(RUN_DIR)/docs.log"; \
	fi

docs-stop: ## Остановить MkDocs
	$(call STOP_SERVICE,docs,$(DOCS_PORT))

docs-serve: install-docs ## MkDocs dev-сервер на переднем плане (:8002)
	$(UV) mkdocs serve -a 127.0.0.1:$(DOCS_PORT)

docs-build: install-docs ## Собрать статический сайт в site/
	$(UV) mkdocs build

$(RUN_DIR):
	@powershell.exe -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(RUN_DIR)' | Out-Null" 2>/dev/null || mkdir -p $(RUN_DIR)

# Stop background service by pid file + free TCP port (Windows-friendly).
ifeq ($(OS),Windows_NT)
define STOP_SERVICE
	@powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/stop_service.ps1 \
		-Name $(1) -Port $(2) -PidFile $(RUN_DIR)/$(1).pid
endef
else
define STOP_SERVICE
	@if [ -f $(RUN_DIR)/$(1).pid ]; then \
		pid=$$(cat $(RUN_DIR)/$(1).pid); \
		if kill -0 $$pid 2>/dev/null; then kill $$pid 2>/dev/null || true; fi; \
		rm -f $(RUN_DIR)/$(1).pid; \
		echo "Остановлен $(1) (pid $$pid)"; \
	fi
	@if command -v fuser >/dev/null 2>&1; then fuser -k $(2)/tcp 2>/dev/null || true; fi
endef
endif
