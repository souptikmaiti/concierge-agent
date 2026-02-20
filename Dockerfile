# syntax=docker/dockerfile:1

FROM python:3.13-slim-trixie

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.7.15 /uv /bin/

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# UV configuration
ENV UV_LINK_MODE=copy \
    PRODUCTION_MODE=true

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Copy source code (needed for editable install)
COPY concierge_agent ./concierge_agent

# Install dependencies
RUN uv sync --no-cache --link-mode copy

# Copy remaining files
COPY . .

# Set environment variables
ENV PRODUCTION_MODE=True \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "--no-sync", "concierge_agent"]
