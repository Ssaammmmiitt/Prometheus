# ---- Frontend Build Stage ----
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Python Backend Stage ----
FROM python:3.11-slim

# Install system dependencies required by geospatial libraries
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PROMETHEUS_FORECASTS_ROOT=/data/forecasts

WORKDIR $HOME/app

# Install API and App dependencies
COPY --chown=user pyproject.toml .
COPY --chown=user BUILD_PLAN.md .
COPY --chown=user src/ ./src/
RUN pip install --no-cache-dir fastapi uvicorn "titiler.core>=0.18.0" && \
    pip install --no-cache-dir -e .

# Copy configs and static data required by the models
COPY --chown=user configs/ ./configs/
COPY --chown=user data/static/ ./data/static/
COPY --chown=user data/models/bundles/ ./data/models/bundles/

# Copy the built frontend
COPY --chown=user --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["uvicorn", "prometheus.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
