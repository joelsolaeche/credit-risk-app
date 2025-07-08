#!/bin/bash
# Startup script for Railway deployment

# Set default port if PORT is not set
if [ -z "$PORT" ]; then
    PORT=8000
fi

echo "Starting uvicorn on port $PORT"
uvicorn app:app --host 0.0.0.0 --port $PORT 