FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -e "."

COPY alembic alembic/
COPY alembic.ini .
COPY app app/
COPY scripts scripts/
COPY livekit.yaml .

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
