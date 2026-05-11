#!/usr/bin/env bash
# ============================================
# Piezo.AI v2.1.0 — Development Utility Script
# ============================================
# Usage: bash scripts/dev.sh <command>
#
# Commands:
#   setup       Incremental setup (keeps existing deps)
#   setup:all   Full clean + fresh install
#   clean       Remove node_modules, .next, __pycache__, .venv
#   db:create   Create the piezo_ai database
#   db:reset    Drop and recreate DB + run migrations
#   db:migrate  Run Alembic migrations
#   start       Start backend + frontend dev servers
#   stop        Gracefully shut down all servers + free ports
# ============================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()     { echo -e "${CYAN}[piezo-ai]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
SETUP_WARNINGS=()
SETUP_ERRORS=()
IS_SETUP_COMMAND=false

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
    SETUP_WARNINGS+=("$1")
}
err() {
    echo -e "${RED}[✗]${NC} $1"
    SETUP_ERRORS+=("$1")
}

# ── Summary on exit (only for setup commands) ──
print_summary() {
    local exit_code=$?
    if [ "$IS_SETUP_COMMAND" = false ]; then return; fi

    echo ""
    echo "=========================================================================="
    if [ $exit_code -eq 0 ] && [ ${#SETUP_ERRORS[@]} -eq 0 ]; then
        if [ ${#SETUP_WARNINGS[@]} -eq 0 ]; then
            echo -e "${GREEN}✅ SETUP COMPLETED SUCCESSFULLY!${NC}"
            echo -e "${GREEN}   All dependencies, database, and migrations are ready.${NC}"
            echo -e "${GREEN}   Run: bash scripts/dev.sh start${NC}"
        else
            echo -e "${YELLOW}⚠️  SETUP COMPLETED WITH WARNINGS:${NC}"
            for w in "${SETUP_WARNINGS[@]}"; do echo -e "   -> $w"; done
            echo -e "${GREEN}   Run: bash scripts/dev.sh start${NC}"
        fi
    else
        echo -e "${RED}❌ SETUP FAILED:${NC}"
        for e in "${SETUP_ERRORS[@]}"; do echo -e "   -> $e"; done
        if [ ${#SETUP_WARNINGS[@]} -gt 0 ]; then
            echo -e "${YELLOW}Warnings:${NC}"
            for w in "${SETUP_WARNINGS[@]}"; do echo -e "   -> $w"; done
        fi
        echo -e "${CYAN}Review errors above, fix, and retry: ${NC}bash scripts/dev.sh setup"
    fi
    echo "=========================================================================="
}
trap print_summary EXIT

# ── Python version: 3.11–3.13 ──
PYTHON_CMD=""
VENV_DIR="$ROOT_DIR/.venv"

find_python() {
    for candidate in python3.13 python3.12 python3.11; do
        local path
        path=$(command -v "$candidate" 2>/dev/null) || continue
        if "$path" --version &>/dev/null; then
            PYTHON_CMD="$path"
            return 0
        fi
    done
    for ver in 3.13 3.12 3.11; do
        for prefix in /opt/homebrew/bin /usr/local/bin; do
            if [ -x "$prefix/python${ver}" ]; then
                PYTHON_CMD="$prefix/python${ver}"
                return 0
            fi
        done
    done
    if command -v python3 &>/dev/null; then
        local minor
        minor=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$minor" -ge 11 ] && [ "$minor" -lt 14 ]; then
            PYTHON_CMD="python3"
            return 0
        fi
    fi
    return 1
}

# ── Ensure Node.js is available (auto-loads nvm if needed) ──
ensure_node() {
    # If node is already on PATH and >= v20, we're good
    if command -v node &>/dev/null; then
        local major
        major=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
        if [ "${major:-0}" -ge 20 ]; then
            return 0
        fi
    fi

    # Try to load nvm
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        log "Loading nvm..."
        export NVM_DIR="$HOME/.nvm"
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
        nvm use 20 2>/dev/null || nvm use default 2>/dev/null || true
    fi

    # Re-check
    if command -v node &>/dev/null; then
        local major
        major=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
        if [ "${major:-0}" -ge 20 ]; then
            return 0
        fi
    fi

    return 1
}

# ── Activate venv ──
activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        return 0
    fi
    return 1
}

# ── DB connection from .env ──
get_db_vars() {
    local raw_url=""
    if [ -f "$ROOT_DIR/.env" ]; then
        raw_url=$(grep -E '^DATABASE_URL=' "$ROOT_DIR/.env" | head -1 | cut -d'=' -f2-)
    fi
    DB_URL="${raw_url:-postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai}"

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

# ── Port collision check ──
check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
        warn "Port $port ($service) is already in use."
        read -p "  Kill the conflicting process? (y/N): " kill_choice
        echo ""
        if [[ "$kill_choice" =~ ^[Yy]$ ]]; then
            kill -9 "$(lsof -Pi :"$port" -sTCP:LISTEN -t)" 2>/dev/null || true
            success "Freed port $port."
        else
            err "Cannot start $service — port $port is occupied."
            exit 1
        fi
    fi
}

# ── Ensure Docker DB is running (used by setup, start, db:* commands) ──
ensure_db_running() {
    get_db_vars

    # Quick check: is DB already reachable?
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        return 0
    fi
    export PGPASSWORD="$DB_PASS"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "SELECT 1;" >/dev/null 2>&1; then
        unset PGPASSWORD
        return 0
    fi
    unset PGPASSWORD

    # DB not reachable — check Docker
    warn "PostgreSQL is not running on $DB_HOST:$DB_PORT."

    if ! command -v docker &>/dev/null; then
        err "Docker is not installed and PostgreSQL is not reachable."
        err "Install Docker Desktop or start PostgreSQL manually on port $DB_PORT."
        return 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        err "Docker is installed but the daemon is not running."
        err "Please start Docker Desktop, then retry."
        return 1
    fi

    # Docker available — try to start/restart container
    log "Attempting to start PostgreSQL via Docker..."
    docker compose -f docker/docker-compose.yml up -d 2>/dev/null ||
    docker-compose -f docker/docker-compose.yml up -d 2>/dev/null || {
        err "Failed to start PostgreSQL container."
        return 1
    }
    success "PostgreSQL container started"

    # Wait for readiness
    log "Waiting for database to be ready..."
    for i in {1..30}; do
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
            success "Database is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    err "Timed out waiting for PostgreSQL (30s). Check: docker logs piezo-ai-db"
    return 1
}

# ======================================================================
# COMMANDS
# ======================================================================

cmd_setup() {
    local force_clean="${1:-}"
    IS_SETUP_COMMAND=true

    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║     Piezo.AI v2.1.0 — Setup          ║"
    echo "  ╚══════════════════════════════════════╝"
    echo -e "${NC}"

    # 0. Optional clean
    if [ "$force_clean" == "all" ]; then
        log "Running full clean before setup..."
        cmd_clean
    fi

    # 1. .env
    log "[Step 1/6] Checking environment variables..."
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            success ".env created from .env.example"
        else
            warn ".env.example not found. Create .env manually."
        fi
    else
        success ".env already exists"
    fi

    # 2. Python
    log "[Step 2/6] Checking Python version (requires 3.11–3.13)..."
    if ! find_python; then
        err "No compatible Python found (need 3.11, 3.12, or 3.13)."
        err "Python 3.14 is NOT supported — mendeleev requires <3.14."
        err "Install: brew install python@3.13"
        return 1
    fi
    local py_version
    py_version=$("$PYTHON_CMD" --version)
    success "Using $py_version ($PYTHON_CMD)"

    # 3. Virtual environment + Python deps
    log "[Step 3/6] Setting up Python virtual environment (.venv/)..."
    if [ ! -d "$VENV_DIR" ]; then
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        success "Virtual environment created at .venv/"
    else
        local venv_minor
        venv_minor=$("$VENV_DIR/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$venv_minor" -lt 11 ] || [ "$venv_minor" -ge 14 ]; then
            warn "Existing .venv uses incompatible Python. Recreating..."
            rm -rf "$VENV_DIR"
            "$PYTHON_CMD" -m venv "$VENV_DIR"
            success "Virtual environment recreated with $py_version"
        else
            success "Virtual environment exists (Python 3.$venv_minor)"
        fi
    fi
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip --quiet 2>/dev/null || warn "Could not upgrade pip (no internet?)"

    log "  Installing packages/db..."
    pip install -e packages/db || { err "Failed to install packages/db"; return 1; }
    success "packages/db installed"

    log "  Installing packages/ml-core..."
    pip install -e packages/ml-core || { err "Failed to install packages/ml-core"; return 1; }
    success "packages/ml-core installed"

    log "  Installing apps/api..."
    pip install -e "apps/api[dev]" || { err "Failed to install apps/api"; return 1; }
    success "apps/api installed"

    # 4. Node.js + frontend
    log "[Step 4/6] Checking Node.js and installing frontend dependencies..."
    if ! ensure_node; then
        warn "Node.js 20+ not found. Frontend will not work."
        warn "Install: nvm install 20 && nvm use 20"
        warn "Or: brew install node@20"
    else
        success "Node.js $(node -v)"

        if ! command -v pnpm &>/dev/null; then
            log "  pnpm not found. Installing via npm..."
            npm install -g pnpm 2>/dev/null || warn "Failed to install pnpm globally"
        fi

        if command -v pnpm &>/dev/null; then
            success "pnpm $(pnpm --version)"
            pnpm install || { err "pnpm install failed"; return 1; }
            success "Frontend dependencies installed"
        else
            warn "pnpm still not available. Install manually: npm i -g pnpm"
        fi
    fi

    # 5. Database
    log "[Step 5/6] Setting up PostgreSQL database..."
    setup_db_environment
    cmd_db_create

    # 6. Migrations
    log "[Step 6/6] Applying database migrations..."
    cmd_db_migrate "--from-setup"

    echo ""
    success "Setup complete! Run: bash scripts/dev.sh start"
}

cmd_clean() {
    log "Cleaning project artifacts..."

    rm -rf node_modules apps/web/node_modules apps/web/.next .turbo .pnpm-store pnpm-lock.yaml
    success "Removed node_modules, .next, .turbo caches"

    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
    rm -rf .venv
    success "Removed __pycache__ and .venv"

    rm -rf apps/api/*.egg-info packages/*/*.egg-info build dist
    success "Removed egg-info and build dirs"

    log "Clean complete. Run: bash scripts/dev.sh setup"
}

setup_db_environment() {
    get_db_vars

    echo ""
    echo -e "${CYAN}--- Database Configuration ---${NC}"
    echo "  1) Docker Container (recommended)"
    echo "  2) Local PostgreSQL installation"
    read -t 20 -p "Select option [1/2] (auto-defaults to 1 in 20s): " db_option || true
    echo ""

    db_option="${db_option:-1}"

    if [[ "$db_option" == "2" ]]; then
        log "Using local PostgreSQL installation."
        if ! ensure_db_running; then
            err "Cannot connect to local PostgreSQL. Ensure it is running."
            return 1
        fi
    else
        log "Using Docker PostgreSQL."
        if ! command -v docker &>/dev/null; then
            err "Docker not installed. Install: https://docker.com/products/docker-desktop"
            return 1
        fi
        if ! docker info &>/dev/null 2>&1; then
            err "Docker daemon not running! Start Docker Desktop first."
            return 1
        fi

        log "Starting PostgreSQL container..."
        docker compose -f docker/docker-compose.yml up -d 2>/dev/null ||
        docker-compose -f docker/docker-compose.yml up -d 2>/dev/null || {
            err "Failed to start PostgreSQL container."
            return 1
        }
        success "PostgreSQL container started"

        log "Waiting for database..."
        for i in {1..30}; do
            if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
                success "Database is ready!"
                return 0
            fi
            echo -n "."
            sleep 1
        done
        echo ""
        warn "Timed out waiting for DB (30s). It may still be starting."
    fi
}

cmd_db_create() {
    ensure_db_running || return 1
    get_db_vars
    export PGPASSWORD="$DB_PASS"

    log "Checking if database '$DB_NAME' exists..."

    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        success "Database '$DB_NAME' exists."
    else
        log "Creating database '$DB_NAME'..."
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null ||
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE \"$DB_NAME\";" 2>/dev/null ||
        docker exec piezo-ai-db psql -U piezo -c "CREATE DATABASE piezo_ai;" 2>/dev/null || {
            err "Failed to create database."
            return 1
        }
        success "Database '$DB_NAME' created."
    fi
    unset PGPASSWORD
}

cmd_db_reset() {
    ensure_db_running || return 1
    get_db_vars

    warn "This will DROP and RECREATE '$DB_NAME'. ALL DATA WILL BE LOST!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Aborted."
        return
    fi

    export PGPASSWORD="$DB_PASS"
    log "Dropping database '$DB_NAME'..."
    dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null ||
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" 2>/dev/null ||
    docker exec piezo-ai-db psql -U piezo -c "DROP DATABASE IF EXISTS piezo_ai;" 2>/dev/null || {
        err "Failed to drop database."
        return 1
    }
    unset PGPASSWORD

    cmd_db_create

    log "Running migrations on fresh database..."
    activate_venv || true
    SYNC_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    DATABASE_URL="$SYNC_URL" alembic -c packages/db/alembic.ini upgrade head || {
        err "Migrations failed after reset. Check schema code."
        return 1
    }
    success "Database hard reset complete."
}

cmd_db_migrate() {
    ensure_db_running || return 1
    get_db_vars
    log "Applying database migrations..."

    activate_venv || true

    SYNC_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')

    if DATABASE_URL="$SYNC_URL" alembic -c packages/db/alembic.ini upgrade head 2>/tmp/piezo_alembic.log; then
        success "Migrations applied successfully."
        rm -f /tmp/piezo_alembic.log
        return 0
    fi

    err "Migration failed!"
    cat /tmp/piezo_alembic.log 2>/dev/null || true
    echo ""

    if [ "${1:-}" != "--from-setup" ]; then
        warn "HARD RESET? This deletes all data in '$DB_NAME'."
        read -p "Type 'yes' to reset, or anything else to abort: " reset_choice
        if [[ "$reset_choice" == "yes" ]]; then
            cmd_db_reset
        else
            err "Aborted. Fix the database manually."
            return 1
        fi
    else
        warn "Migration failed during setup. Try: bash scripts/dev.sh db:reset"
    fi
}

cmd_start() {
    log "Starting Piezo.AI v2.1.0 development servers..."

    # Ensure DB is running
    if ! ensure_db_running; then
        err "Database is not available. Cannot start."
        exit 1
    fi
    success "Database is active and ready."

    # Port checks
    check_port 8000 "Backend (FastAPI)"
    check_port 3000 "Frontend (Next.js)"

    # Activate venv
    if ! activate_venv; then
        err "Virtual environment not found. Run: bash scripts/dev.sh setup"
        exit 1
    fi

    # Start backend
    log "Starting FastAPI backend on port 8000..."
    cd "$ROOT_DIR/apps/api"
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload \
        --reload-dir "$ROOT_DIR/apps/api" \
        --reload-dir "$ROOT_DIR/packages" &
    BACKEND_PID=$!
    cd "$ROOT_DIR"

    # Start frontend
    FRONTEND_PID=""
    if ensure_node; then
        if ! command -v pnpm &>/dev/null; then
            warn "pnpm not found. Trying npm install -g pnpm..."
            npm install -g pnpm 2>/dev/null || true
        fi

        if command -v pnpm &>/dev/null; then
            if [ ! -d "apps/web/node_modules" ] || [ ! -d "node_modules" ]; then
                log "Missing node_modules. Running pnpm install..."
                pnpm install || warn "pnpm install failed"
            fi
            log "Starting Next.js frontend on port 3000..."
            pnpm dev:web &
            FRONTEND_PID=$!
        else
            warn "pnpm not available. Frontend will not start."
        fi
    else
        warn "Node.js 20+ not found. Frontend will not start."
        warn "Fix: nvm install 20 && nvm use 20 && npm i -g pnpm"
    fi

    # Summary
    echo ""
    echo -e "${CYAN}${BOLD}════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  Piezo.AI v2.1.0 is running!${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}Frontend:${NC}  http://localhost:3000"
    echo -e "  ${BOLD}Backend:${NC}   http://localhost:8000"
    echo -e "  ${BOLD}API Docs:${NC}  http://localhost:8000/docs"
    echo -e "  ${BOLD}Health:${NC}    http://localhost:8000/health"
    echo ""
    echo -e "  Press ${BOLD}Ctrl+C${NC} to stop all services."

    # ── Single Ctrl+C shutdown handler ──
    SHUTDOWN_IN_PROGRESS=false

    shutdown_all() {
        if [ "$SHUTDOWN_IN_PROGRESS" = true ]; then return; fi
        SHUTDOWN_IN_PROGRESS=true

        echo -e "\n\n${YELLOW}Shutting down Piezo.AI v2.1.0...${NC}"

        if [ -n "${BACKEND_PID:-}" ]; then
            kill "$BACKEND_PID" 2>/dev/null
            sleep 1
            kill -0 "$BACKEND_PID" 2>/dev/null && kill -9 "$BACKEND_PID" 2>/dev/null
            echo -e "  ${GREEN}✓${NC} Backend stopped (PID: $BACKEND_PID)"
        fi

        if [ -n "${FRONTEND_PID:-}" ]; then
            kill "$FRONTEND_PID" 2>/dev/null
            sleep 1
            kill -0 "$FRONTEND_PID" 2>/dev/null && kill -9 "$FRONTEND_PID" 2>/dev/null
            echo -e "  ${GREEN}✓${NC} Frontend stopped (PID: $FRONTEND_PID)"
        else
            echo -e "  ${YELLOW}-${NC} Frontend was not running"
        fi

        # Fallback: kill anything still on ports
        for port in 8000 3000; do
            if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                kill -9 "$(lsof -Pi :"$port" -sTCP:LISTEN -t)" 2>/dev/null || true
                echo -e "  ${GREEN}✓${NC} Cleaned up stale process on port $port"
            fi
        done

        # Stop Docker DB if using Docker
        if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
            docker compose -f docker/docker-compose.yml stop 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} Docker PostgreSQL container stopped"
        fi

        echo -e "${GREEN}All services stopped cleanly.${NC}"
        exit 0
    }

    trap shutdown_all INT TERM
    wait
}

cmd_stop() {
    log "Gracefully shutting down Piezo.AI v2.1.0..."

    for port in 8000 3000; do
        local label="Backend"
        [ "$port" = "3000" ] && label="Frontend"
        if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
            local pids
            pids=$(lsof -Pi :"$port" -sTCP:LISTEN -t)
            kill -9 $pids 2>/dev/null || true
            success "$label stopped (port $port, PID: $pids)"
        else
            success "$label already stopped (port $port free)"
        fi
    done

    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        log "Stopping Docker PostgreSQL container..."
        docker compose -f docker/docker-compose.yml stop 2>/dev/null || true
        success "PostgreSQL container stopped"
    fi

    success "Environment shut down."
}

# ======================================================================
# MAIN
# ======================================================================
case "${1:-}" in
    setup)      cmd_setup ;;
    setup:all)  cmd_setup "all" ;;
    clean)      cmd_clean ;;
    db:create)  cmd_db_create ;;
    db:reset)   cmd_db_reset ;;
    db:migrate) cmd_db_migrate ;;
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    *)
        echo -e "${CYAN}${BOLD}Piezo.AI v2.1.0 — Development Tool${NC}"
        echo ""
        echo "Usage: bash scripts/dev.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup       Incremental setup (keeps existing deps if compatible)"
        echo "  setup:all   Full clean + fresh install (wipes .venv, node_modules)"
        echo "  clean       Remove node_modules, .next, __pycache__, .venv"
        echo "  db:create   Create the database"
        echo "  db:reset    Drop and recreate DB + run migrations"
        echo "  db:migrate  Run Alembic migrations only"
        echo "  start       Start backend + frontend dev servers"
        echo "  stop        Gracefully shut down all servers + free ports"
        echo ""
        echo "Python: 3.11–3.13 required (3.14 not supported by mendeleev)"
        echo "Node:   20+ required"
        ;;
esac
