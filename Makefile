.PHONY: all tui install test lint clean

# Default: just type `make` and the TUI starts
all: tui

tui:
	openclaw tui

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -q

lint:
	python -m ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
