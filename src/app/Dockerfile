# Use the official Python image as the base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9 as base

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install the required dependencies for the app
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY ./ /app

FROM base as build

# Expose the port that will be used by Railway
EXPOSE $PORT

# Health check for the web service using Python requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import requests; requests.get('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

# Run the main FastAPI app when the container starts
ENTRYPOINT ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
