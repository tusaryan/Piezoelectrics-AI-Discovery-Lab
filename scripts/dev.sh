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

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
        else
            echo -e "${YELLOW}⚠️  SETUP COMPLETED WITH WARNINGS:${NC}"
            for w in "${SETUP_WARNINGS[@]}"; do echo -e "   -> $w"; done
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

# ── Required Python version: 3.11–3.13 ──
# mendeleev requires <3.14, so we must use 3.11, 3.12, or 3.13
REQUIRED_PYTHON_MIN="3.11"
REQUIRED_PYTHON_MAX="3.14"  # exclusive upper bound
PYTHON_CMD=""
VENV_DIR="$ROOT_DIR/.venv"

# Find a compatible Python (prefer python3.13, then 3.12, then 3.11)
find_python() {
    for candidate in python3.13 python3.12 python3.11; do
        local path
        path=$(command -v "$candidate" 2>/dev/null) || continue
        # Check if it works
        if "$path" --version &>/dev/null; then
            PYTHON_CMD="$path"
            return 0
        fi
    done

    # Try homebrew paths directly (macOS)
    for ver in 3.13 3.12 3.11; do
        local brew_path="/opt/homebrew/bin/python${ver}"
        if [ -x "$brew_path" ]; then
            PYTHON_CMD="$brew_path"
            return 0
        fi
        brew_path="/usr/local/bin/python${ver}"
        if [ -x "$brew_path" ]; then
            PYTHON_CMD="$brew_path"
            return 0
        fi
    done

    # Fallback: check if `python3` is within range
    if command -v python3 &>/dev/null; then
        local ver
        ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -lt 14 ]; then
            PYTHON_CMD="python3"
            return 0
        fi
    fi

    return 1
}

# ── DB connection from .env ──
get_db_vars() {
    if [ -f "$ROOT_DIR/.env" ]; then
        DB_URL=$(grep -E '^DATABASE_URL=' "$ROOT_DIR/.env" | head -1 | cut -d'=' -f2-)
    fi
    DB_URL="${DB_URL:-postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai}"

    CLEAN_URL=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|')
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
            err "Cannot start $service — port $port is occupied. Free it manually."
            exit 1
        fi
    fi
}

# ── Ensure PostgreSQL is reachable before any DB command ──
ensure_db_running() {
    get_db_vars

    # Quick connectivity check via pg_isready or psql
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        return 0
    fi

    # Also try direct psql (in case pg_isready is not installed)
    export PGPASSWORD="$DB_PASS"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "SELECT 1;" >/dev/null 2>&1; then
        unset PGPASSWORD
        return 0
    fi
    unset PGPASSWORD

    # DB is not reachable — try Docker
    warn "PostgreSQL is not running on $DB_HOST:$DB_PORT."
    echo ""

    if ! command -v docker &>/dev/null; then
        err "Docker is not installed and PostgreSQL is not reachable."
        err "Either install Docker Desktop or start PostgreSQL manually on port $DB_PORT."
        return 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        err "Docker is installed but the daemon is not running."
        err "Please start Docker Desktop, then retry."
        return 1
    fi

    # Docker is available — offer to start the container
    echo -e "${CYAN}Would you like to start the PostgreSQL Docker container?${NC}"
    read -p "  Start Docker PostgreSQL? (Y/n): " start_choice
    echo ""

    if [[ "$start_choice" =~ ^[Nn]$ ]]; then
        err "Cannot proceed without a running database. Start PostgreSQL and retry."
        return 1
    fi

    log "Starting PostgreSQL container..."
    if docker compose -f docker/docker-compose.yml up -d 2>/dev/null; then
        success "PostgreSQL container started"
    elif docker-compose -f docker/docker-compose.yml up -d 2>/dev/null; then
        success "PostgreSQL container started (legacy compose)"
    else
        err "Failed to start PostgreSQL container. Check docker/docker-compose.yml."
        return 1
    fi

    # Wait for readiness
    log "Waiting for database to be ready..."
    for i in {1..30}; do
        if docker exec piezo-ai-db pg_isready -U piezo -d piezo_ai &>/dev/null; then
            success "Database is ready!"
            return 0
        fi
        # Also check via host connection
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
            success "Database is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""

    err "Timed out waiting for PostgreSQL (30s). Check Docker logs: docker logs piezo-ai-db"
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
        cp .env.example .env
        success ".env created from .env.example"
    else
        success ".env already exists"
    fi

    # 2. Find compatible Python
    log "[Step 2/6] Checking Python version (requires 3.11–3.13)..."
    if ! find_python; then
        err "No compatible Python found (need 3.11, 3.12, or 3.13)."
        err "Python 3.14 is NOT supported — mendeleev requires <3.14."
        err "Install Python 3.13: brew install python@3.13"
        exit 1
    fi
    local py_version
    py_version=$("$PYTHON_CMD" --version)
    success "Using $py_version ($PYTHON_CMD)"

    # 3. Create/update virtual environment
    log "[Step 3/6] Setting up Python virtual environment (.venv/)..."
    if [ ! -d "$VENV_DIR" ]; then
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        success "Virtual environment created at .venv/"
    else
        # Verify the venv Python is compatible
        local venv_ver
        venv_ver=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        local venv_minor
        venv_minor=$(echo "$venv_ver" | cut -d. -f2)
        if [ "$venv_minor" -lt 11 ] || [ "$venv_minor" -ge 14 ]; then
            warn "Existing .venv uses Python $venv_ver (incompatible). Recreating..."
            rm -rf "$VENV_DIR"
            "$PYTHON_CMD" -m venv "$VENV_DIR"
            success "Virtual environment recreated with $py_version"
        else
            success "Virtual environment exists (Python $venv_ver)"
        fi
    fi
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip --quiet 2>/dev/null || warn "Could not upgrade pip (no internet?)"

    # Install Python packages
    log "  Installing packages/db..."
    pip install -e packages/db || { err "Failed to install packages/db"; exit 1; }
    success "packages/db installed"

    log "  Installing packages/ml-core..."
    pip install -e packages/ml-core || { err "Failed to install packages/ml-core"; exit 1; }
    success "packages/ml-core installed"

    log "  Installing apps/api..."
    pip install -e "apps/api[dev]" || { err "Failed to install apps/api"; exit 1; }
    success "apps/api installed"

    # 4. Node.js + pnpm
    log "[Step 4/6] Checking Node.js and installing frontend dependencies..."
    if ! command -v node &>/dev/null; then
        warn "Node.js not found. Frontend will not work."
        warn "Install Node 20+: nvm install 20 && nvm use 20"
    else
        local node_major
        node_major=$(node -v | sed 's/v//' | cut -d. -f1)
        if [ "$node_major" -lt 20 ]; then
            warn "Node.js $(node -v) found but v20+ required. Run: nvm install 20 && nvm use 20"
        else
            success "Node.js $(node -v)"

            if ! command -v pnpm &>/dev/null; then
                log "  pnpm not found. Installing via npm..."
                npm install -g pnpm || warn "Failed to install pnpm globally"
            fi

            if command -v pnpm &>/dev/null; then
                success "pnpm $(pnpm --version)"
                pnpm install || { err "pnpm install failed"; exit 1; }
                success "Frontend dependencies installed"
            else
                warn "pnpm still not available. Install manually: npm i -g pnpm"
            fi
        fi
    fi

    # 5. Database setup
    log "[Step 5/6] Setting up PostgreSQL database..."
    setup_db_environment
    cmd_db_create

    # 6. Run migrations
    log "[Step 6/6] Applying database migrations..."
    cmd_db_migrate "--from-setup"

    echo ""
    success "Setup complete! Run: bash scripts/dev.sh start"
}

cmd_clean() {
    log "Cleaning project artifacts..."

    # Node
    rm -rf node_modules apps/web/node_modules apps/web/.next .turbo .pnpm-store pnpm-lock.yaml
    success "Removed node_modules, .next, .turbo caches"

    # Python
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
    rm -rf .venv
    success "Removed __pycache__ and .venv"

    # Build artifacts
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
    read -t 10 -p "Select option [1/2] (auto-defaults to 1 in 10s): " db_option || true
    echo ""

    db_option="${db_option:-1}"

    if [[ "$db_option" == "2" ]]; then
        log "Using local PostgreSQL installation."
    else
        log "Using Docker PostgreSQL."
        if ! command -v docker &>/dev/null; then
            err "Docker not installed. Install: https://docker.com/products/docker-desktop"
            exit 1
        fi
        if ! docker info &>/dev/null 2>&1; then
            err "Docker daemon not running! Start Docker Desktop first."
            exit 1
        fi

        log "Starting PostgreSQL container..."
        if docker compose -f docker/docker-compose.yml up -d 2>/dev/null; then
            success "PostgreSQL container started"
        elif docker-compose -f docker/docker-compose.yml up -d 2>/dev/null; then
            success "PostgreSQL container started (legacy compose)"
        else
            err "Failed to start PostgreSQL container."
            exit 1
        fi

        # Wait for DB readiness
        log "Waiting for database..."
        for i in {1..20}; do
            if docker exec piezo-ai-db pg_isready -U piezo -d piezo_ai &>/dev/null; then
                success "Database is ready!"
                return 0
            fi
            echo -n "."
            sleep 1
            if [ "$i" -eq 20 ]; then
                warn "Timed out waiting for DB. It may still be starting."
            fi
        done
    fi
}

cmd_db_create() {
    ensure_db_running || exit 1
    get_db_vars
    export PGPASSWORD="$DB_PASS"

    log "Checking if database '$DB_NAME' exists..."

    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        success "Database '$DB_NAME' exists."
    else
        log "Creating database '$DB_NAME'..."
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null || \
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE \"$DB_NAME\";" 2>/dev/null || \
        docker exec piezo-ai-db psql -U piezo -c "CREATE DATABASE piezo_ai;" 2>/dev/null || \
        { err "Failed to create database."; exit 1; }
        success "Database '$DB_NAME' created."
    fi
    unset PGPASSWORD
}

cmd_db_reset() {
    ensure_db_running || exit 1
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
    dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" 2>/dev/null || \
    docker exec piezo-ai-db psql -U piezo -c "DROP DATABASE IF EXISTS piezo_ai;" 2>/dev/null || \
    { err "Failed to drop database."; exit 1; }
    unset PGPASSWORD

    cmd_db_create

    log "Running migrations on fresh database..."
    [ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"
    DATABASE_URL="$DB_URL" alembic -c packages/db/alembic.ini upgrade head || {
        err "Migrations failed after reset. Check schema code."
        exit 1
    }
    success "Database hard reset complete."
}

cmd_db_migrate() {
    ensure_db_running || exit 1
    get_db_vars
    log "Applying database migrations..."

    [ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

    # Auto-generate initial migration if versions/ is empty
    local versions_dir="$ROOT_DIR/packages/db/alembic/versions"
    local has_migrations
    has_migrations=$(find "$versions_dir" -name '*.py' 2>/dev/null | head -1)
    if [ -z "$has_migrations" ]; then
        log "No migration revisions found. Auto-generating initial migration..."
        DATABASE_URL="$DB_URL" alembic -c packages/db/alembic.ini revision --autogenerate -m "initial_schema" 2>/tmp/piezo_alembic.log || {
            err "Failed to generate initial migration."
            cat /tmp/piezo_alembic.log 2>/dev/null || true
            return 1
        }
        success "Initial migration generated."
    fi

    if DATABASE_URL="$DB_URL" alembic -c packages/db/alembic.ini upgrade head 2>/tmp/piezo_alembic.log; then
        success "Migrations applied successfully."
        rm -f /tmp/piezo_alembic.log
        return 0
    fi

    err "Migration failed!"
    cat /tmp/piezo_alembic.log 2>/dev/null || true
    echo ""

    # Only prompt for hard reset if NOT called from setup
    if [ "${1:-}" != "--from-setup" ]; then
        warn "HARD RESET? This deletes all data in '$DB_NAME'."
        read -p "Type 'yes' to reset, or anything else to abort: " reset_choice
        if [[ "$reset_choice" == "yes" ]]; then
            cmd_db_reset
        else
            err "Aborted. Fix the database manually."
            exit 1
        fi
    else
        warn "Migration failed during setup. DB may need manual intervention."
    fi
}

cmd_start() {
    log "Starting Piezo.AI v2.1.0 development servers..."

    # Check DB
    get_db_vars
    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        log "Ensuring PostgreSQL container is running..."
        docker compose -f docker/docker-compose.yml up -d 2>/dev/null || true
        sleep 2
    fi

    # Port checks
    check_port 8000 "Backend (FastAPI)"
    check_port 3000 "Frontend (Next.js)"

    # Activate venv
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        err "Virtual environment not found. Run: bash scripts/dev.sh setup"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"

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
    if command -v node &>/dev/null && command -v pnpm &>/dev/null; then
        local node_major
        node_major=$(node -v | sed 's/v//' | cut -d. -f1)
        if [ "$node_major" -ge 20 ]; then
            # Ensure deps exist
            if [ ! -d "apps/web/node_modules" ] || [ ! -d "node_modules" ]; then
                log "Missing node_modules. Running pnpm install..."
                pnpm install || warn "pnpm install failed"
            fi
            log "Starting Next.js frontend on port 3000..."
            pnpm dev:web &
            FRONTEND_PID=$!
        else
            warn "Node $(node -v) found but v20+ required. Skipping frontend."
        fi
    else
        warn "Node.js or pnpm not found. Frontend will not start."
        warn "Install: nvm install 20 && nvm use 20 && npm i -g pnpm"
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
    echo -e "  ${BOLD}Database:${NC}  postgresql://piezo:piezo@localhost:5432/piezo_ai"
    echo -e ""
    echo -e "  Press ${BOLD}Ctrl+C${NC} to stop all services."

    # ── Single Ctrl+C shutdown handler ──
    # Guard flag prevents double execution (trap fires on INT, then script exits)
    SHUTDOWN_IN_PROGRESS=false

    shutdown_all() {
        if [ "$SHUTDOWN_IN_PROGRESS" = true ]; then return; fi
        SHUTDOWN_IN_PROGRESS=true

        echo -e "\n\n${YELLOW}Shutting down Piezo.AI v2.1.0...${NC}"

        # 1. Kill backend by PID
        if [ -n "${BACKEND_PID:-}" ]; then
            kill "$BACKEND_PID" 2>/dev/null
            # Wait briefly for graceful shutdown
            sleep 1
            # Force kill if still alive
            kill -0 "$BACKEND_PID" 2>/dev/null && kill -9 "$BACKEND_PID" 2>/dev/null
            echo -e "  ${GREEN}✓${NC} Backend stopped"
        fi

        # 2. Kill frontend by PID
        if [ -n "${FRONTEND_PID:-}" ]; then
            kill "$FRONTEND_PID" 2>/dev/null
            sleep 1
            kill -0 "$FRONTEND_PID" 2>/dev/null && kill -9 "$FRONTEND_PID" 2>/dev/null
            echo -e "  ${GREEN}✓${NC} Frontend stopped"
        fi

        # 3. Fallback: kill anything still on ports 8000/3000 (catches child processes)
        for port in 8000 3000; do
            if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                kill -9 "$(lsof -Pi :"$port" -sTCP:LISTEN -t)" 2>/dev/null || true
            fi
        done

        # 4. Stop Docker PostgreSQL container
        if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
            docker compose -f docker/docker-compose.yml stop 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} PostgreSQL container stopped"
        fi

        echo -e "${GREEN}All services stopped cleanly.${NC}"
        exit 0
    }

    # Only trap INT and TERM — NOT EXIT (avoids double execution)
    trap shutdown_all INT TERM

    # Wait for background processes (will be interrupted by Ctrl+C → trap fires)
    wait
}

cmd_stop() {
    log "Gracefully shutting down Piezo.AI v2.1.0..."

    # Kill processes on known ports (PID-based via lsof)
    for port in 8000 3000; do
        if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
            local pids
            pids=$(lsof -Pi :"$port" -sTCP:LISTEN -t)
            kill -9 $pids 2>/dev/null || true
            success "Freed port $port (killed PID: $pids)"
        else
            success "Port $port already free"
        fi
    done

    # Stop Docker container
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
