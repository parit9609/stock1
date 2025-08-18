.PHONY: help install install-dev test test-cov lint format clean build docs run-demo run-validation

# Default target
help:
	@echo "Stock Market Prediction - Production Commands"
	@echo "============================================="
	@echo ""
	@echo "Installation:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test             Run all tests"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  lint             Run linting (flake8, mypy)"
	@echo "  format           Format code (black, isort)"
	@echo ""
	@echo "Development:"
	@echo "  run-demo         Run consistent validation demo"
	@echo "  run-validation   Run comprehensive evaluation"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  docs             Build documentation"
	@echo ""
	@echo "Example: make install-dev test lint format"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=stock_prediction --cov-report=html --cov-report=term-missing

# Code Quality
lint:
	flake8 stock_prediction/ tests/
	mypy stock_prediction/

format:
	black stock_prediction/ tests/
	isort stock_prediction/ tests/

# Development
run-demo:
	python -m stock_prediction.scripts.consistent_validation_demo --config config/training_config.yaml

run-validation:
	python -m stock_prediction.scripts.comprehensive_evaluation --config config/training_config.yaml

# Build & Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build: clean
	python -m build

docs:
	cd docs && make html

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Quick validation
validate: lint test
	@echo "✅ All validation checks passed!"

# Production deployment check
deploy-check: clean install test lint format
	@echo "✅ Production deployment checks passed!"
	@echo "Ready for deployment!"

# Development workflow
dev-workflow: install-dev format lint test
	@echo "✅ Development workflow completed successfully!"
