# Copyright (c) Meta Platforms, Inc.
# Licensed under BSD-style license.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install git (required for dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Build args
ARG BUILD_MODE=in-repo
ARG ENV_NAME=hospital_env

# Copy project files
COPY . /app/env

WORKDIR /app/env

# Install uv if missing
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync; \
    else \
        uv sync; \
    fi


# -----------------------------
# 🚀 RUNTIME STAGE
# -----------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Install curl in runtime stage for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /app/env/.venv /app/.venv

# Copy app code
COPY --from=builder /app/env /app/env

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check (important for HF Spaces)
HEALTHCHECK CMD curl -f http://localhost:7860/ || exit 1

# ✅ CORRECT ENTRYPOINT (IMPORTANT FIX)
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 7860"]