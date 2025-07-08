# Use the official Python image as the base image
FROM python:3.11-slim as base

# Set the working directory in the container
WORKDIR /src

# Copy requirements first for better Docker layer caching
COPY requirements.txt /src/requirements.txt

# Install the required dependencies for the model
RUN pip install --upgrade pip && pip install --no-cache-dir -r /src/requirements.txt

# Copy all files and directories from the models directory into the container
COPY ./ /src

FROM base as build

# Health check for the model service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import redis; r=redis.from_url('$REDIS_URL') if '$REDIS_URL' else redis.Redis(host='redis'); r.ping()" || exit 1

# Run the model script when the container starts
ENTRYPOINT ["python3", "ml_model.py"]
