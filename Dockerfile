FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY gsea_refiner/ gsea_refiner/
COPY scripts/ scripts/
COPY data/example/ data/example/
COPY data/config/ data/config/
COPY data/training/ data/training/
COPY data/gold/ data/gold/

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "scripts/benchmark.py"]
