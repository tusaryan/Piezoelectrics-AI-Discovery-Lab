#!/usr/bin/env bash
# ============================================
# Piezo.AI v2.1.0 — Development Utility Script
# ============================================
# Usage: bash scripts/dev.sh <command>
#
# Commands:
#   setup       Incremental setup (keeps existing deps if compatible)
#   setup:all   Full clean + fresh install (wipes .venv, node_modules)
#   clean       Remove node_modules, .next, __pycache__, .venv
#   db:create   Create the piezo_ai database
#   db:reset    Drop and recreate DB + run migrations
#   db:migrate  Run Alembic migrations only
#   start       Start backend + frontend dev servers
#   stop        Gracefully shut down all servers + free ports
#
# Library modules (scripts/lib/):
#   _colors.sh   — Colors, logging, summary printer
#   _python.sh   — Python/venv detection and pip install helpers
#   _node.sh     — Node.js / pnpm detection
#   _database.sh — PostgreSQL (Docker + local), migrations
#   _network.sh  — Connectivity diagnostics
# ============================================

set -uo pipefail

# ── Bootstrap ────────────────────────────────
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR="$(cd "$SOURCE_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Source all library modules
for lib in "$SOURCE_DIR/lib/"_*.sh; do
    # shellcheck source=SCRIPTDIR/lib/_colors.sh
    source "$lib"
done

IS_SETUP_COMMAND=false

# ════════════════════════════════════════════
# COMMAND: setup
# ════════════════════════════════════════════
cmd_setup() {
    local force_clean="${1:-}"
    IS_SETUP_COMMAND=true

    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║     ${APP_NAME:-Piezo.AI} v${APP_VERSION:-2.1.0} — Setup          ║"
    echo "  ╚══════════════════════════════════════╝"
    echo -e "${NC}"

    # 0. Optional clean
    if [ "$force_clean" == "all" ]; then
        pz_log "Running full clean..."
        cmd_clean
    fi

    # 1. .env
    pz_log "[Step 1/7] Checking environment variables..."
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            pz_success ".env created from .env.example"
        else
            pz_warn ".env.example not found. Create .env manually."
        fi
    else
        pz_success ".env already exists"
    fi

    # 2. Python
    pz_log "[Step 2/7] Checking Python version (requires 3.11–3.13)..."
    if ! pz_find_python; then
        pz_err "No compatible Python found (need 3.11, 3.12, or 3.13)."
        pz_err "Python 3.14 is NOT supported — mendeleev requires <3.14."
        pz_err "Install: brew install python@3.13"
        return 1
    fi
    pz_success "Using $_PZ_PYTHON_VERSION ($_PZ_PYTHON_CMD)"

    # 3. Network check before pip install
    pz_log "[Step 3/7] Checking network connectivity..."
    if ! pz_check_pypi; then
        pz_warn "Cannot reach pypi.org — package downloads may fail."
        pz_diagnose_network
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            pz_log "Aborted."
            return 1
        fi
        pz_warn "Proceeding with limited network — retry if install fails."
    else
        pz_success "pypi.org is reachable"
    fi

    # 4. Virtual environment + Python deps
    pz_log "[Step 4/7] Setting up Python virtual environment..."
    pz_activate_venv 2>/dev/null || true

    if ! pz_ensure_venv; then
        pz_err "Failed to create virtual environment"
        return 1
    fi

    # Upgrade pip first (fixes build dependencies)
    pz_log "  Upgrading pip..."
    source "$_PZ_VENV_DIR/bin/activate"
    if pip install --upgrade pip --quiet 2>&1 | tail -2; then
        pz_success "pip upgraded"
    else
        pz_warn "pip upgrade failed (network may be slow)"
    fi
    source "$VENV_DIR/bin/activate"

    # Install build dependencies (fixes setuptools >=75.0 requirement)
    pz_log "  Ensuring build dependencies..."
    if pip install 'setuptools>=75.0' 'wheel' --quiet 2>&1 | tail -2; then
        pz_success "build dependencies ready"
    else
        pz_warn "Could not install build dependencies — will retry per-package"
    fi

    # Install packages
    local install_fail=false

    pz_log "[Step 5/7] Installing Python packages..."
    pz_activate_venv || { pz_err "venv activation failed"; return 1; }

    pz_pip_install "packages/db" "" || install_fail=true
    pz_pip_install "packages/ml-core" "[symbolic]" || install_fail=true
    pz_pip_install "apps/api" "[dev]" || install_fail=true

    if [ "$install_fail" = true ]; then
        pz_warn "One or more packages failed to install."
        pz_warn "Common causes: network issues, missing system libs (libomp, gfortran)."
        pz_warn "Heavy deps (pymatgen, xgboost, lightgbm) may need extra build time."
        pz_warn "Retry: bash scripts/dev.sh setup"
        pz_info "TIP: For faster retry, use: pip install --no-build-isolation ..."
    fi

    # 5. Node.js + frontend
    pz_log "[Step 6/7] Setting up frontend..."
    if pz_ensure_node; then
        pz_ensure_pnpm || pz_warn "pnpm unavailable"
        pz_pnpm_install || pz_warn "pnpm install failed — frontend deps may be missing"
    fi

    # 6. Database
    pz_log "[Step 7/7] Setting up database..."
    pz_db_setup_interactive
    pz_db_create
    pz_log "Running Alembic migrations..."
    pz_db_migrate "--from-setup"

    echo ""
    pz_success "Setup complete! Run: bash scripts/dev.sh start"
}

# ════════════════════════════════════════════
# COMMAND: clean
# ════════════════════════════════════════════
cmd_clean() {
    pz_log "Removing project artifacts..."

    rm -rf node_modules apps/web/node_modules apps/web/.next .turbo .pnpm-store pnpm-lock.yaml
    pz_success "Removed node_modules, .next, .turbo caches"

    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
    rm -rf .venv
    pz_success "Removed __pycache__ and .venv"

    rm -rf apps/api/*.egg-info packages/*/*.egg-info build dist
    pz_success "Removed egg-info and build dirs"

    pz_log "Clean complete. Run: bash scripts/dev.sh setup"
}

# ════════════════════════════════════════════
# COMMAND: db:create
# ════════════════════════════════════════════
cmd_db_create() {
    IS_SETUP_COMMAND=true
    pz_db_create
}

# ════════════════════════════════════════════
# COMMAND: db:migrate
# ════════════════════════════════════════════
cmd_db_migrate() {
    IS_SETUP_COMMAND=true
    pz_activate_venv 2>/dev/null || true
    pz_db_migrate
}

# ════════════════════════════════════════════
# COMMAND: db:reset
# ════════════════════════════════════════════
cmd_db_reset() {
    IS_SETUP_COMMAND=true
    pz_activate_venv 2>/dev/null || true
    pz_db_reset
}

# ════════════════════════════════════════════
# COMMAND: start
# ════════════════════════════════════════════
cmd_start() {
    pz_log "Starting ${APP_NAME:-Piezo.AI} v${APP_VERSION:-2.1.0}..."

    # Ensure DB
    if ! pz_ensure_db_running; then
        pz_err "Database not available. Cannot start."
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
    pz_success "Database is active"

    # Port checks
    pz_log "Checking ports..."
    for port in 8000 3000; do
        if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
            local label="Backend"
            [ "$port" = "3000" ] && label="Frontend"
            pz_warn "Port $port ($label) is already in use"
            read -p "  Kill conflicting process? (y/N): " kill_choice
            echo ""
            if [[ "$kill_choice" =~ ^[Yy]$ ]]; then
                kill -9 "$(lsof -Pi :"$port" -sTCP:LISTEN -t)" 2>/dev/null || true
                pz_success "Freed port $port"
            else
                pz_err "Cannot start — port $port is occupied"
                return 1
            fi
        fi
    done

    # venv
    if ! pz_activate_venv; then
        pz_err "Virtual environment not found. Run: bash scripts/dev.sh setup"
        return 1
    fi

    # Start backend
    pz_log "Starting FastAPI backend on port 8000..."
    cd "$ROOT_DIR/apps/api"
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload \
        --reload-dir "$ROOT_DIR/apps/api" \
        --reload-dir "$ROOT_DIR/packages" &
    BACKEND_PID=$!
    cd "$ROOT_DIR"

    # Start frontend
    FRONTEND_PID=""
    if pz_ensure_node; then
        pz_ensure_pnpm
        pz_pnpm_install
        if command -v pnpm &>/dev/null; then
            pz_log "Starting Next.js frontend on port 3000..."
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
    echo -e "${GREEN}${BOLD}  ${APP_NAME:-Piezo.AI} v${APP_VERSION:-2.1.0} is running!${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}Frontend:${NC}  http://localhost:3000"
    echo -e "  ${BOLD}Backend:${NC}   http://localhost:8000"
    echo -e "  ${BOLD}API Docs:${NC}  http://localhost:8000/docs"
    echo -e "  ${BOLD}Health:${NC}    http://localhost:8000/health"
    echo ""
    echo -e "  Press ${BOLD}Ctrl+C${NC} to stop all services."

    # ── Graceful shutdown ─────────────────────
    SHUTDOWN_IN_PROGRESS=false

    shutdown_all() {
        if [ "$SHUTDOWN_IN_PROGRESS" = true ]; then return; fi
        SHUTDOWN_IN_PROGRESS=true
        echo -e "\n\n${YELLOW}Shutting down...${NC}"

        for pid_var in BACKEND_PID FRONTEND_PID; do
            local pid="${!pid_var}"
            if [ -n "$pid" ]; then
                kill "$pid" 2>/dev/null
                sleep 1
                kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
                echo -e "  ${GREEN}✓${NC} Stopped PID $pid"
            fi
        done

        # Cleanup stale processes on ports
        for port in 8000 3000; do
            if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                kill -9 "$(lsof -Pi :"$port" -sTCP:LISTEN -t)" 2>/dev/null || true
                echo -e "  ${GREEN}✓${NC} Cleaned up stale process on port $port"
            fi
        done

        pz_db_stop_docker
        echo -e "${GREEN}All services stopped.${NC}"
        exit 0
    }

    trap shutdown_all INT TERM
    wait
}

# ════════════════════════════════════════════
# COMMAND: stop
# ════════════════════════════════════════════
cmd_stop() {
    pz_log "Shutting down..."

    for port in 8000 3000; do
        local label="Backend"
        [ "$port" = "3000" ] && label="Frontend"
        if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
            local pids
            pids=$(lsof -Pi :"$port" -sTCP:LISTEN -t)
            kill -9 $pids 2>/dev/null || true
            pz_success "$label stopped (port $port)"
        else
            pz_success "$label already stopped"
        fi
    done

    pz_db_stop_docker
    pz_success "Environment shut down."
}

# ════════════════════════════════════════════
# COMMAND: diagnose
# ════════════════════════════════════════════
cmd_diagnose() {
    pz_log "Running full diagnostics..."

    pz_log "--- Python ---"
    if pz_find_python; then
        pz_success "Python: $_PZ_PYTHON_VERSION ($_PZ_PYTHON_CMD)"
    else
        pz_err "Python: not found or incompatible"
    fi

    pz_log "--- Network ---"
    pz_diagnose_network

    pz_log "--- pip ---"
    if pz_activate_venv 2>/dev/null && pip --version; then
        pz_success "pip ready"
    else
        pz_warn "pip not available in venv"
    fi

    pz_log "--- Node.js ---"
    if pz_ensure_node; then
        pz_success "Node.js: $NODE_VERSION"
    fi

    pz_log "--- Database ---"
    pz_get_db_vars
    if pz_db_is_ready; then
        pz_success "Database: reachable ($DB_HOST:$DB_PORT/$DB_NAME)"
    else
        pz_warn "Database: not reachable"
    fi
}

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
trap pz_print_summary EXIT

case "${1:-}" in
    setup)        cmd_setup ;;
    setup:all)    cmd_setup "all" ;;
    clean)        cmd_clean ;;
    db:create)    cmd_db_create ;;
    db:reset)     cmd_db_reset ;;
    db:migrate)   cmd_db_migrate ;;
    start)        cmd_start ;;
    stop)         cmd_stop ;;
    diagnose)     cmd_diagnose ;;
    *)
        echo -e "${CYAN}${BOLD}${APP_NAME:-Piezo.AI} v${APP_VERSION:-2.1.0} — Dev Tool${NC}"
        echo ""
        echo "Usage: bash scripts/dev.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup       Incremental setup (keeps existing deps if compatible)"
        echo "  setup:all   Full clean + fresh install"
        echo "  clean       Remove node_modules, .next, __pycache__, .venv"
        echo "  db:create   Create the database"
        echo "  db:reset    Drop and recreate DB + run migrations"
        echo "  db:migrate  Run Alembic migrations only"
        echo "  start       Start backend + frontend dev servers"
        echo "  stop        Shut down all servers + Docker DB"
        echo "  diagnose    Run full diagnostics (Python, network, DB, Node)"
        echo ""
        echo "Python: 3.11–3.13 required | Node: 20+ required"
        ;;
esac
