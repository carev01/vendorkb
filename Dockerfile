# syntax=docker/dockerfile:1.7
# One image for every vendorkb role; the role is chosen by the container command.
# Contains NO secret: configuration arrives as env vars from the cluster.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/carev01/vendorkb"
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
USER 10001
CMD ["vendorkb", "sync"]
