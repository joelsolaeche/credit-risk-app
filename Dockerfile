# Multi-stage build for Railway deployment
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Copy web app requirements first for better caching
COPY src/app/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the web application files
COPY src/app/ /app/

# Copy the model files needed by the web service
COPY src/models/predict_model.pkl /app/predict_model.pkl

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

# Start the web application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"] 