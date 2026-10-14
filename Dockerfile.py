# EdgeDrive: Automotive AI & Edge Intelligence Platform
# Multi-stage Dockerfile for edge deployment

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

LABEL maintainer="EdgeDrive Team"
LABEL description="Automotive AI perception platform"
LABEL version="1.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Set Python path
ENV PYTHONPATH=/app/src:/app:$PYTHONPATH
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command: start API server
CMD ["python", "-c", "from edgedrive.api.server import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"]