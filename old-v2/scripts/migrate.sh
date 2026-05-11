#!/bin/bash
set -e

echo "Running Alembic migrations..."
cd packages/db
alembic upgrade head
echo "Migrations complete!"
