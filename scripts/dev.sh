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
    
    # 4. Setup Database Environment & Create DB
    log "[Step 4/5] Preparing Database Environment..."
    setup_db_environment
    cmd_db_create
    
    # 5. Run migrations
    log "[Step 5/5] Applying Database Migrations..."
    cmd_db_migrate "--from-setup"
    
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

setup_db_environment() {
    get_db_vars
    echo ""
    echo -e "${CYAN}--- Database Configuration ---${NC}"
    echo "How would you like to run the PostgreSQL database?"
    echo "  1) Docker Container (Default - Recommended)"
    echo "  2) Local Installation (Use existing postgres service)"
    
    # Read with 10 sec timeout, default to 1
    read -t 10 -p "Select option [1/2] (Auto-defaults to 1 in 10s): " db_option || true
    echo ""
    
    db_option="${db_option:-1}"
    
    if [[ "$db_option" == "2" ]]; then
        log "Using Local PostgreSQL Installation."
    else
        log "Using Docker PostgreSQL."
        if ! command -v docker &> /dev/null; then
            err "Docker is not installed. Falling back to Local assumption."
        elif ! docker info &> /dev/null; then
            err "Docker daemon is not running! Please start Docker Desktop/daemon."
            exit 1
        else
            log "Checking Docker database container state..."
            
            if ! docker compose -f docker/docker-compose.dev.yml up -d db 2>/dev/null && ! docker-compose -f docker/docker-compose.dev.yml up -d db 2>/dev/null; then
                warn "Failed to start Docker container normally. Attempting hard reset of the container..."
                docker compose -f docker/docker-compose.dev.yml down -v db 2>/dev/null || true
                docker compose -f docker/docker-compose.dev.yml up -d db || { err "Failed to recover Docker container. Ensure ports are open and image is not corrupted."; exit 1; }
            fi
            
            # Explicitly unpause and start if they were left in a limbo state
            docker compose -f docker/docker-compose.dev.yml unpause db 2>/dev/null || true
            docker compose -f docker/docker-compose.dev.yml start db 2>/dev/null || true
            
            log "Waiting for Docker Database to be ready..."
            for i in {1..30}; do
                if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
                    success "Docker Database is ready!"
                    break
                fi
                sleep 1
                if [ $i -eq 30 ]; then
                    warn "Timed out waiting for Docker DB. It might still be starting."
                fi
            done
        fi
    fi
}

cmd_db_create() {
    get_db_vars
    export PGPASSWORD="$DB_PASS"
    
    log "Checking if database '$DB_NAME' exists on $DB_HOST:$DB_PORT..."
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        success "Database '$DB_NAME' is present."
    else
        log "Database '$DB_NAME' not found. Creating..."
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null || \
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE \"$DB_NAME\";" 2>/dev/null || \
        { err "Failed to create database. Is PostgreSQL running?"; exit 1; }
        success "Database '$DB_NAME' created successfully."
    fi
    unset PGPASSWORD
}

cmd_db_reset() {
    local force=$1
    get_db_vars
    
    if [[ "$force" != "--force" ]]; then
        warn "This will DROP and RECREATE the '$DB_NAME' database. All data will be lost!"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Aborted."
            return
        fi
    fi
    
    export PGPASSWORD="$DB_PASS"
    
    log "Dropping database '$DB_NAME'..."
    dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" 2>/dev/null || \
    { err "Failed to drop database."; exit 1; }
    
    unset PGPASSWORD
    
    cmd_db_create
    
    log "Running migrations on fresh database..."
    if [ -f "apps/api/.venv/bin/activate" ]; then
        source apps/api/.venv/bin/activate
    fi
    SYNC_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    DATABASE_URL="$SYNC_URL" alembic -c packages/db/alembic.ini upgrade head || {
        err "CRITICAL: Migrations failed even after a fresh reset. There is a fundamental code or schema error."
        exit 1
    }
    success "Database hard reset complete."
}

cmd_db_migrate() {
    get_db_vars
    log "Applying database migrations..."
    
    if [ -f "apps/api/.venv/bin/activate" ]; then
        source apps/api/.venv/bin/activate
    fi
    
    # Ensure DB is created if user calls `bash scripts/dev.sh db:migrate` standalone
    export PGPASSWORD="$DB_PASS"
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log "Database does not exist yet. Creating before migration..."
        cmd_db_create
    fi
    unset PGPASSWORD
    
    SYNC_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    
    # Try migrating
    if DATABASE_URL="$SYNC_URL" alembic -c packages/db/alembic.ini upgrade head 2>/tmp/piezo_alembic.log; then
        success "Database migrations applied successfully. Database is initialized and ready."
        rm -f /tmp/piezo_alembic.log
        return 0
    fi
    
    # If we reached here, migration failed.
    err "Migration failed! The database might be corrupted, incompatible, or not initialized properly."
    log "Error Details:"
    cat /tmp/piezo_alembic.log || true
    echo ""
    
    warn "Would you like to perform a HARD RESET? This will COMPLETELY DELETE the existing '$DB_NAME' database, recreate it, and rerun migrations. ALL DATA WILL BE LOST!"
    read -p "Type 'yes' to Hard Reset, or any other key to Abort: " reset_choice
    echo ""
    
    if [[ "$reset_choice" == "yes" ]]; then
        log "User initiated Hard Reset. Proceeding..."
        cmd_db_reset "--force"
    else
        err "Migration aborted by user. Please fix the database state manually or drop it to reset."
        exit 1
    fi
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
    
    log "Checking if database is running securely before starting..."
    get_db_vars
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        warn "PostgreSQL does not seem to be running/responding on $DB_HOST:$DB_PORT."
        if command -v docker &> /dev/null && docker info &> /dev/null; then
             log "Docker is available. Attempting to revive container if it is paused or exited..."
             docker compose -f docker/docker-compose.dev.yml unpause db 2>/dev/null || true
             docker compose -f docker/docker-compose.dev.yml start db 2>/dev/null || true
             docker compose -f docker/docker-compose.dev.yml up -d db 2>/dev/null || true
             sleep 3
        fi
    else
        success "Database is active and ready."
    fi

    # Port Collision Checks
    check_port() {
        local port=$1
        local service=$2
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            warn "Port $port ($service) is already in use by another process."
            read -p "Would you like to automatically stop the conflicting process to continue? (y/N): " kill_choice
            echo ""
            if [[ "$kill_choice" =~ ^[Yy]$ ]]; then
                log "Killing process on port $port..."
                kill -9 $(lsof -Pi :$port -sTCP:LISTEN -t) 2>/dev/null || true
                success "Freed port $port."
            else
                err "Cannot start $service. Please free port $port manually and try again."
                exit 1
            fi
        fi
    }

    check_port 8000 "Backend"
    check_port 3000 "Frontend"

    # Backend
    if [ -f "apps/api/.venv/bin/activate" ]; then
        source apps/api/.venv/bin/activate
    fi
    
    log "Starting backend on port 8000..."
    uvicorn apps.api.app.main:app --reload --port 8000 &
    BACKEND_PID=$!
    
    # Frontend
    log "Verifying frontend dependencies..."
    if [ ! -d "node_modules" ] || [ ! -d "apps/web/node_modules" ]; then
        warn "Missing node_modules detected. Running pnpm install automatically to recover dependencies..."
        if command -v pnpm &> /dev/null; then
            pnpm install || { err "pnpm install failed"; exit 1; }
        else
            warn "pnpm not found natively. Attempting npm install -g pnpm to fix..."
            npm install -g pnpm 2>/dev/null || true
            pnpm install || { err "pnpm installation failed"; exit 1; }
        fi
        success "Dependencies forcefully recovered successfully!"
    fi

    log "Starting frontend on port 3000..."
    cd apps/web && pnpm run dev &
    FRONTEND_PID=$!
    cd "$ROOT_DIR"
    
    success "Backend PID: $BACKEND_PID | Frontend PID: $FRONTEND_PID"
    log "Press Ctrl+C to gracefully stop the servers and clean ports"
    
    trap "log 'Caught interrupt signal! Initiating clean shutdown...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; cmd_stop bypass; exit" INT TERM
    wait
}

cmd_stop() {
    log "Gracefully shutting down Piezo.AI v2 environment..."

    # Kill Background Server PIDs explicitly via lsof if they are still holding ports
    log "Cleaning up stranded local processes on ports 8000 & 3000..."
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        kill -9 $(lsof -Pi :8000 -sTCP:LISTEN -t) 2>/dev/null || true
    fi
    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        kill -9 $(lsof -Pi :3000 -sTCP:LISTEN -t) 2>/dev/null || true
    fi

    # Docker cleanup
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        log "Halting Docker database container to save resources..."
        docker compose -f docker/docker-compose.dev.yml stop db 2>/dev/null || docker-compose -f docker/docker-compose.dev.yml stop db 2>/dev/null || true
    fi
    
    success "Environment cleanly shut down."
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
    stop)       cmd_stop ;;
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
        echo "  stop        Gracefully shut down all servers, free ports, and sleep Docker"
        ;;
esac
