#!/bin/bash
# ============================================
# Piezo.AI v2 — Development Utility Script
# ============================================
# Usage: bash scripts/dev.sh <command>
#
# Commands:
#   setup       Full setup (deps + DB + migrations)
#   clean       Remove node_modules, .next, __pycache__, .venv
#   db:create   Create the piezo_ai database
#   db:reset    Drop and recreate DB + run migrations
#   db:migrate  Run Alembic migrations
#   db:seed     Seed sample datasets from resources/
#   start       Start both backend & frontend

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() { echo -e "${CYAN}[piezo-ai]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }

# ------- Detect DB connection from .env -------
get_db_vars() {
    if [ -f "$ROOT_DIR/.env" ]; then
        DB_URL=$(grep -E '^DATABASE_URL=' "$ROOT_DIR/.env" | head -1 | cut -d'=' -f2-)
    fi
    DB_URL="${DB_URL:-postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai}"
    
    # Extract parts from URL (strip asyncpg driver prefix)
    CLEAN_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    DB_USER=$(echo "$CLEAN_URL" | sed -n 's|postgresql://\([^:]*\):.*|\1|p')
    DB_PASS=$(echo "$CLEAN_URL" | sed -n 's|postgresql://[^:]*:\([^@]*\)@.*|\1|p')
    DB_HOST=$(echo "$CLEAN_URL" | sed -n 's|.*@\([^:]*\):.*|\1|p')
    DB_PORT=$(echo "$CLEAN_URL" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
    DB_NAME=$(echo "$CLEAN_URL" | sed -n 's|.*/\(.*\)|\1|p')
    
    DB_USER="${DB_USER:-piezo}"
    DB_PASS="${DB_PASS:-piezo}"
    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"
    DB_NAME="${DB_NAME:-piezo_ai}"
}

# ------- Commands -------

cmd_setup() {
    local force_clean="$1"

    log "Setting up Piezo.AI v2 Dev Environment..."
    
    # 0. Clean existing environment if requested
    if [ "$force_clean" == "all" ]; then
        log "Running clean before setup to ensure fresh start..."
        cmd_clean
    else
        log "Incremental setup (skipping clean). Run 'setup:all' for a fresh install."
    fi
    
    # 1. Copy .env if needed
    log "[Step 1/5] Checking environment variables..."
    if [ ! -f .env ]; then
        log "Copying .env.example to .env..."
        cp .env.example .env
        success ".env created"
    else
        success ".env already exists"
    fi
    
    # 2. Install JS dependencies
    log "[Step 2/5] Installing Node.js dependencies (pnpm)..."
    if ! command -v pnpm &> /dev/null; then
        log "pnpm not found. Installing pnpm globally via npm..."
        npm install -g pnpm
    fi
    
    if command -v pnpm &> /dev/null; then
        pnpm install
    else
        err "Failed to install pnpm. Please install it manually: npm i -g pnpm"
        exit 1
    fi
    success "Node.js dependencies installed"
    
    # 3. Setup Python venv
    log "[Step 3/5] Setting up Python virtual environment (pip)..."
    if [ ! -d "apps/api/.venv" ]; then
        python3 -m venv apps/api/.venv
    fi
    source apps/api/.venv/bin/activate
    
    # We remove -q so pip shows its native download/install progress bars
    pip install -e apps/api
    pip install -e packages/ml-core
    pip install -e packages/db
    success "Python dependencies installed"
    
    # 4. Create DB
    log "[Step 4/5] Preparing Database..."
    cmd_db_create
    
    # 5. Run migrations
    log "[Step 5/5] Applying Database Migrations..."
    cmd_db_migrate
    
    success "Setup complete! Run 'bash scripts/dev.sh start' to launch."
}

cmd_clean() {
    log "Cleaning project artifacts..."
    
    # Node
    rm -rf node_modules apps/web/node_modules apps/web/.next
    success "Removed node_modules and .next"
    
    # Python
    find apps packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    rm -rf apps/api/.venv
    success "Removed __pycache__ and .venv"
    
    # Build artifacts
    rm -rf apps/api/api.egg-info packages/*/dist packages/*/*.egg-info
    success "Removed egg-info and dist directories"
    
    log "Clean complete. Run 'bash scripts/dev.sh setup' to reinstall."
}

cmd_db_create() {
    get_db_vars
    log "Creating database '$DB_NAME' on $DB_HOST:$DB_PORT..."
    
    export PGPASSWORD="$DB_PASS"
    
    # Check if DB exists
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        success "Database '$DB_NAME' already exists"
    else
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null || \
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || \
        { err "Failed to create database. Make sure PostgreSQL is running and user '$DB_USER' exists."; exit 1; }
        success "Database '$DB_NAME' created"
    fi
    
    unset PGPASSWORD
}

cmd_db_reset() {
    get_db_vars
    warn "This will DROP and RECREATE the '$DB_NAME' database. All data will be lost!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Aborted."
        return
    fi
    
    export PGPASSWORD="$DB_PASS"
    
    log "Dropping database '$DB_NAME'..."
    dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || \
    { err "Failed to drop database."; exit 1; }
    
    unset PGPASSWORD
    
    cmd_db_create
    cmd_db_migrate
    
    success "Database reset complete."
}

cmd_db_migrate() {
    get_db_vars
    log "Running Alembic migrations..."
    
    # Activate venv if exists
    if [ -f "apps/api/.venv/bin/activate" ]; then
        source apps/api/.venv/bin/activate
    fi
    
    # Convert async URL to sync for Alembic
    SYNC_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    
    DATABASE_URL="$SYNC_URL" alembic -c packages/db/alembic.ini upgrade head 2>/dev/null || \
    DATABASE_URL="$SYNC_URL" python -m alembic -c packages/db/alembic.ini upgrade head || \
    { err "Alembic migration failed. Check your DATABASE_URL and that alembic is installed."; exit 1; }
    
    success "Migrations applied"
}

cmd_db_seed() {
    log "Seeding sample datasets..."
    
    if [ -f "apps/api/.venv/bin/activate" ]; then
        source apps/api/.venv/bin/activate
    fi
    
    python3 -c "
import sys
sys.path.insert(0, '.')
print('[seed] Sample datasets available in resources/')
print('  - resources/sample_knn_basic.csv (basic d33/tc)')
print('  - resources/sample_knn_pvdf_composite.csv (PVDF composite)')
print('  - resources/sample_hardness_only.csv (hardness)')
print('[seed] Upload these via the web UI Dataset page.')
"
    
    success "Seed info displayed. Use the web UI to upload datasets."
}

cmd_start() {
    log "Starting Piezo.AI v2 development servers..."
    
    # Backend
    if [ -f "apps/api/.venv/bin/activate" ]; then
        source apps/api/.venv/bin/activate
    fi
    
    log "Starting backend on port 8000..."
    uvicorn apps.api.app.main:app --reload --port 8000 &
    BACKEND_PID=$!
    
    # Frontend
    log "Starting frontend on port 3000..."
    cd apps/web && npm run dev &
    FRONTEND_PID=$!
    cd "$ROOT_DIR"
    
    success "Backend PID: $BACKEND_PID | Frontend PID: $FRONTEND_PID"
    log "Press Ctrl+C to stop both servers"
    
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
    wait
}

# ------- Main -------
case "${1:-}" in
    setup)      cmd_setup ;;
    setup:all)  cmd_setup "all" ;;
    clean)      cmd_clean ;;
    db:create)  cmd_db_create ;;
    db:reset)   cmd_db_reset ;;
    db:migrate) cmd_db_migrate ;;
    db:seed)    cmd_db_seed ;;
    start)      cmd_start ;;
    *)
        echo "Piezo.AI v2 Development Tool"
        echo ""
        echo "Usage: bash scripts/dev.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup       Incremental setup (keeps existing dependencies)"
        echo "  setup:all   Full clean and reinstall (wipes everything first)"
        echo "  clean       Remove node_modules, .next, __pycache__, .venv"
        echo "  db:create   Create the database"
        echo "  db:reset    Drop and recreate DB + run migrations"
        echo "  db:migrate  Run Alembic migrations only"
        echo "  db:seed     Show info about sample datasets"
        echo "  start       Start backend + frontend dev servers"
        ;;
esac
