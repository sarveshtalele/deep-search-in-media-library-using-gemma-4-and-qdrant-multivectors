.PHONY: help setup model qdrant api web app ingest search stats test lint fmt clean

help:
	@echo "Deep Search in Media Library — common tasks"
	@echo "  make setup    Create uv venv and install (editable + dev)"
	@echo "  make model    Pull the Gemma 4 + embedding models via Ollama"
	@echo "  make qdrant   (Optional) run Qdrant in Docker instead of local mode"
	@echo "  make api      Launch the FastAPI backend (port 8000)"
	@echo "  make web      Install + launch the Next.js frontend (port 3000)"
	@echo "  make ingest P=path [C=category]   Ingest a file/directory"
	@echo "  make search Q='your query'        One-off CLI search"
	@echo "  make stats    Show collection stats"
	@echo "  make test     Run the test suite (stub backend, no weights needed)"
	@echo "  make lint     Ruff + mypy"

setup:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

model:
	ollama pull gemma4:e4b
	ollama pull nomic-embed-text

qdrant:
	docker run -p 6333:6333 -p 6334:6334 -v $$(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

api:
	uv run uvicorn deepsearch.api.main:app --host 127.0.0.1 --port 8000 --reload

web:
	cd web && npm install && npm run dev

app: ## run backend + frontend together (needs two terminals; this starts the API)
	@echo "Start the API:  make api"
	@echo "Start the web:  make web   (http://localhost:3000)"

ingest:
	uv run deepsearch ingest "$(P)" --category "$(or $(C),uncategorized)"

search:
	uv run deepsearch search "$(Q)"

stats:
	uv run deepsearch stats

test:
	DEEPSEARCH_USE_STUB=true uv run pytest

lint:
	uv run ruff check src tests
	uv run mypy src

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

clean:
	rm -rf qdrant_storage data/cache/* .pytest_cache .ruff_cache .mypy_cache
