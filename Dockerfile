# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for network & C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy GPU/CUDA dependencies in container
RUN pip install --no-cache-dir "torch>=2.2.0" --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project manifest and install core, server, and symbolic verifier dependencies
# (quoted constraints prevent sh from interpreting '>=' as output redirection).
COPY pyproject.toml /app/
RUN pip install --no-cache-dir \
    "transformers>=4.48.0" \
    "datasets>=2.18.0" \
    "accelerate>=0.27.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.22.0" \
    "httpx>=0.25.0" \
    "pydantic>=2.0.0" \
    "pint>=0.23" \
    "sympy>=1.12" \
    "z3-solver>=4.12.0"

# Copy System2 application source code
COPY sys1 /app/sys1
COPY sys2 /app/sys2
COPY routing /app/routing
COPY bert.py /app/bert.py
COPY system.py /app/system.py
COPY encoding.py /app/encoding.py
COPY heads.py /app/heads.py
COPY lora.py /app/lora.py
COPY load_system2.py /app/load_system2.py
COPY README.md /app/README.md

# Prepare /app/models mount point aligned with docker-compose.yml (./models:/app/models:ro).
# Model checkpoints live outside the image layer to keep builds lean; mount ./models
# at runtime (or copy *.pt into /app/models in a downstream image).
RUN mkdir -p /app/models
VOLUME ["/app/models"]

# Install package in editable/console-script mode with extras (--no-deps keeps CPU torch)
RUN pip install --no-cache-dir --no-deps -e ".[server,symbolic]"

EXPOSE 8000

# Default command: run System2 high-speed decision server on CPU
CMD ["sys2-server", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]
