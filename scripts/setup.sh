#!/bin/bash
set -e

echo "Setting up Piezo.AI v2 Dev Environment..."

echo "1. Installing dependencies..."
pnpm install

if [ ! -f .env ]; then
  echo "2. Copying .env.example to .env..."
  cp .env.example .env
fi

echo "3. Starting Docker services..."
docker compose -f docker/docker-compose.dev.yml up -d

echo "Dev environment starting up. Wait a moment then run: bash scripts/migrate.sh"
