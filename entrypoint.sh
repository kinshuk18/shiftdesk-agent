#!/bin/sh
# entrypoint.sh — run once per container start:
#   1. Apply schema (idempotent: CREATE TABLE IF NOT EXISTS, etc.)
#   2. Run seed (idempotent: INSERT ... ON CONFLICT DO NOTHING)
#   3. Start the API server
set -e

echo "==> Applying schema..."
psql "$DATABASE_URL" -f /app/db/schema.sql

echo "==> Running seed..."
python /app/db/seed.py

echo "==> Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
