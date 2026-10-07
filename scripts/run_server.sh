#!/bin/bash
# ExoSphere API server startup script

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Default values
HOST=${EXOSPHERE_HOST:-0.0.0.0}
PORT=${EXOSPHERE_PORT:-8000}
WORKERS=${EXOSPHERE_WORKERS:-1}

echo "Starting ExoSphere API server on ${HOST}:${PORT}"

# Initialize database
python -c "from exosphere.api.db import init_db; import asyncio; asyncio.run(init_db())"

# Start server
exec uvicorn exosphere.api.app:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info