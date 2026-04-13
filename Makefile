.PHONY: dev test migrate seed

dev:
	docker compose up -d && alembic upgrade head && uvicorn app.main:app --reload --reload-exclude .venv --reload-exclude tests --reload-exclude frontend/node_modules --port 8001

test:
	pytest tests/ -v --cov=app --cov-report=html

migrate:
	alembic upgrade head

seed:
	python scripts/seed_demo.py
