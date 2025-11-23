ENV_FILE := .env
-include $(ENV_FILE)

PYTHON ?= $(PYTHON_BIN)
TEST_PYTHONPATH ?=

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available make targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?##"} {printf "\033[32m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install/upgrade project dependencies using python3.12
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

.PHONY: test
test: ## Run pytest with python3.12 (adds TEST_PYTHONPATH when provided)
	@if [ -n "$(TEST_PYTHONPATH)" ]; then \
		PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m pytest; \
	else \
		$(PYTHON) -m pytest; \
	fi

.PHONY: run
run: ## Execute the CLI entrypoint with python3.12
	$(PYTHON) main.py $(ARGS)

.PHONY: clean
clean: ## Remove Python cache artifacts
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
