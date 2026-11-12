# ==============================================================================
# SimuStruct Digital Twin Platform — Production Multi-Stage Container Image
# Stage 1: Build React Engineering Dashboard (Vite + TypeScript)
# Stage 2: Python Runtime with PyTorch, Scikit-Learn & FastAPI Single-Origin Service
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Frontend Build Environment
# ------------------------------------------------------------------------------
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy dependencies manifest and install
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build optimized production bundle
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Final Production Application Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies (build-essential, libgomp1 for OpenMP/PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python backend dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy Backend Application Code & Frozen Certified Model Checkpoints
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/

# Copy Compiled React SPA from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose Service Port
EXPOSE 8000

# Container Healthcheck via FastAPI /api/v1/health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production Entrypoint via Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

