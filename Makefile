.PHONY: install dev-install lint test up down train logs

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	mypy src

test:
	pytest

up:
	docker compose up -d mlflow api dashboard

down:
	docker compose down

train:
	docker compose --profile train up pipeline

logs:
	docker compose logs -f
