# Makefile

.PHONY: help install dev test test-unit test-integration lint format check clean ingest query

help:
	@echo ""
	@echo "RAG Chatbot Platform"
	@echo "====================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

install: ## Install all dependencies
	poetry install

dev: ## Start API server (hot reload)
	ENVIRONMENT=dev uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

serve: ## Start API server (production mode)
	ENVIRONMENT=prod uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

test: ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	pytest tests/integration/ -v --tb=short

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Format with black
	black src/ tests/

check: lint ## Run all quality checks

ingest: ## Ingest documents from a local directory (DIR=path/to/docs)
	ENVIRONMENT=$(ENV) python -c "\
	from src.ingestion.pipeline import IngestionPipeline; \
	p = IngestionPipeline.from_config(); \
	s = p.run_from_directory('$(DIR)'); \
	print(s.summary())"

query: ## Send a test query to local API
	curl -s -X POST http://localhost:8000/chat \
	  -H "Content-Type: application/json" \
	  -d '{"query":"What is the schema of the users table?","stream":false}' \
	  | python3 -m json.tool

health: ## Check API health
	curl -s http://localhost:8000/health | python3 -m json.tool

clean: ## Remove caches and generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage 2>/dev/null || true
