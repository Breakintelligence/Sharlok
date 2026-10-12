# Sharlok setup and run targets.
#   make setup        install Ollama + uv, start Ollama, pull models, install Python deps
#   make run          start the web app at http://localhost:8000
#   make build-index  rebuild the FAISS knowledge store and exit
# Run `make help` for the full list.

.DEFAULT_GOAL := help
.PHONY: help setup install-uv install-ollama deps start-ollama models rerank run build-index

UNAME_S := $(shell uname -s)
UV = $(or $(shell command -v uv 2>/dev/null),$(HOME)/.local/bin/uv)

OLLAMA_HOST ?= http://localhost:11434
Sharlok_MODEL ?= gemma4:latest
Sharlok_VISION_MODEL ?= gemma4:latest
Sharlok_EMBED_MODEL ?= nomic-embed-text
Sharlok_RERANK_MODEL ?= cross-encoder/ms-marco-MiniLM-L-6-v2
OLLAMA_MODELS = $(sort $(Sharlok_MODEL) $(Sharlok_VISION_MODEL) $(Sharlok_EMBED_MODEL))

help:
	@echo "Sharlok targets:"
	@echo "  make setup        Install Ollama and uv, start Ollama, pull models, install Python deps"
	@echo "  make run          Start Sharlok at http://localhost:8000"
	@echo "  make build-index  Rebuild the FAISS knowledge store and exit"
	@echo "  make models       Pull the Ollama models only"
	@echo "  make rerank       Download the reranker model only"

setup: install-uv install-ollama deps start-ollama models rerank
	@echo "Setup complete. Start the app with: make run"

install-uv:
	@if command -v uv >/dev/null 2>&1 || [ -x "$(UV)" ]; then echo "uv already installed."; \
	elif [ "$(UNAME_S)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then brew install uv; \
	else curl -LsSf https://astral.sh/uv/install.sh | sh; fi

install-ollama:
ifeq ($(UNAME_S),Darwin)
	@if command -v ollama >/dev/null 2>&1; then echo "Ollama CLI already installed."; \
	elif command -v brew >/dev/null 2>&1; then brew install ollama; \
	else echo "Homebrew not found. Install Ollama from https://ollama.com/download and re-run make setup."; exit 1; fi
else ifeq ($(UNAME_S),Linux)
	@if command -v ollama >/dev/null 2>&1; then echo "Ollama CLI already installed."; \
	else curl -fsSL https://ollama.com/install.sh | sh; fi
else ifneq ($(filter MINGW% MSYS% CYGWIN%,$(UNAME_S)),)
	@if command -v ollama >/dev/null 2>&1; then echo "Ollama CLI already installed."; \
	else winget install --id Ollama.Ollama -e; fi
else
	@echo "Unsupported OS: $(UNAME_S)"; exit 1
endif

deps: install-uv
	$(UV) sync

start-ollama:
	@if curl -fsS "$(OLLAMA_HOST)/api/tags" >/dev/null 2>&1; then echo "Ollama is running."; \
	else \
	  echo "Starting Ollama..."; \
	  nohup ollama serve >/dev/null 2>&1 & \
	  i=0; until curl -fsS "$(OLLAMA_HOST)/api/tags" >/dev/null 2>&1; do \
	    i=$$((i+1)); [ $$i -ge 30 ] && { echo "Ollama did not start."; exit 1; }; sleep 1; \
	  done; \
	  echo "Ollama is running."; \
	fi

models: start-ollama
	@for model in $(OLLAMA_MODELS); do \
	  echo "Pulling Ollama model $$model..."; \
	  ollama pull "$$model" || exit 1; \
	done

rerank: deps
	@echo "Downloading reranker $(Sharlok_RERANK_MODEL)..."
	$(UV) run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('$(Sharlok_RERANK_MODEL)')"

run:
	$(UV) run main.py

build-index:
	$(UV) run main.py --build-index
