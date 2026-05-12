#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Database
# ============================================

# ── Load .env into environment ───────────────
pz_load_env() {
    if [ -f "$ROOT_DIR/.env" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]]; then continue; fi
            key=$(echo "$line" | cut -d '=' -f 1 | xargs)
            if [ -n "$key" ] && [ -z "${!key+x}" ]; then
                value=$(echo "$line" | cut -d '=' -f 2-)
                value="${value%\"}"
                value="${value#\"}"
                export "$key"="$value"
            fi
        done < "$ROOT_DIR/.env"
    fi
}

# ── Parse DB vars from DATABASE_URL ─────────
pz_get_db_vars() {
    pz_load_env
    local raw_url="${DATABASE_URL:-postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai}"
    DB_URL="$raw_url"

    local clean_url
    clean_url=$(echo "$raw_url" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    DB_USER=$(echo "$clean_url" | sed -n 's|postgresql://\([^:]*\):.*|\1|p')
    DB_PASS=$(echo "$clean_url" | sed -n 's|postgresql://[^:]*:\([^@]*\)@.*|\1|p')
    DB_HOST=$(echo "$clean_url" | sed -n 's|.*@\([^:]*\):.*|\1|p')
    DB_PORT=$(echo "$clean_url" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
    DB_NAME=$(echo "$clean_url" | sed -n 's|.*/\(.*\)|\1|p')

    DB_USER="${DB_USER:-piezo}"
    DB_PASS="${DB_PASS:-piezo}"
    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"
    DB_NAME="${DB_NAME:-piezo_ai}"
}

# ── Check if PostgreSQL is reachable ─────────
pz_db_is_ready() {
    pz_get_db_vars
    export PGPASSWORD="$DB_PASS"
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        unset PGPASSWORD
        return 0
    fi
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "SELECT 1;" >/dev/null 2>&1; then
        unset PGPASSWORD
        return 0
    fi
    unset PGPASSWORD
    return 1
}

# ── Start PostgreSQL via Docker ───────────────
pz_db_start_docker() {
    pz_log "Attempting to start PostgreSQL via Docker..."
    if docker compose -f "$ROOT_DIR/docker/docker-compose.yml" up -d 2>/dev/null || \
       docker-compose -f "$ROOT_DIR/docker/docker-compose.yml" up -d 2>/dev/null; then
        pz_success "PostgreSQL container started"
        return 0
    fi
    pz_err "Failed to start PostgreSQL container"
    return 1
}

# ── Ensure DB is running (docker or local) ──
pz_ensure_db_running() {
    if pz_db_is_ready; then
        pz_success "Database is already running and ready"
        return 0
    fi

    pz_warn "PostgreSQL is not running on $DB_HOST:$DB_PORT"

    if ! command -v docker &>/dev/null; then
        pz_err "Docker is not installed and PostgreSQL is not reachable."
        pz_err "Install Docker Desktop or start PostgreSQL manually."
        return 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        pz_err "Docker daemon is not running. Start Docker Desktop first."
        return 1
    fi

    pz_db_start_docker || return 1

    # Wait for readiness
    pz_log "Waiting for database to be ready..."
    for i in {1..30}; do
        if pz_db_is_ready; then
            pz_success "Database is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    pz_err "Timed out waiting for PostgreSQL (30s). Check: docker logs piezo-ai-db"
    return 1
}

# ── Create the database if it doesn't exist ─
pz_db_create() {
    pz_ensure_db_running || return 1
    pz_get_db_vars
    export PGPASSWORD="$DB_PASS"

    pz_log "Checking if database '$DB_NAME' exists..."
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt 2>/dev/null | \
       cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        pz_success "Database '$DB_NAME' already exists"
    else
        pz_log "Creating database '$DB_NAME'..."
        createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null || \
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE \"$DB_NAME\";" 2>/dev/null || \
        docker exec piezo-ai-db psql -U piezo -c "CREATE DATABASE piezo_ai;" 2>/dev/null || {
            pz_err "Failed to create database"
            unset PGPASSWORD
            return 1
        }
        pz_success "Database '$DB_NAME' created"
    fi
    unset PGPASSWORD
    return 0
}

# ── Run Alembic migrations ───────────────────
pz_db_migrate() {
    pz_ensure_db_running || return 1
    pz_get_db_vars
    source "$_PZ_VENV_DIR/bin/activate" || return 1

    local sync_url
    sync_url=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')

    pz_log "Running Alembic migrations..."
    if DATABASE_URL="$sync_url" alembic -c "$ROOT_DIR/packages/db/alembic.ini" upgrade head 2>&1; then
        pz_success "Migrations applied successfully"
        return 0
    fi

    pz_err "Migration failed!"
    if [ "${1:-}" != "--from-setup" ]; then
        pz_warn "Try hard reset? This deletes ALL data in '$DB_NAME'."
        read -p "Type 'yes' to reset, or anything else to abort: " reset_choice
        if [[ "$reset_choice" == "yes" ]]; then
            pz_db_reset
            return $?
        fi
    else
        pz_warn "Retry with: bash scripts/dev.sh db:reset"
    fi
    return 1
}

# ── Drop and recreate database ───────────────
pz_db_reset() {
    pz_ensure_db_running || return 1
    pz_get_db_vars

    pz_warn "This will DROP and RECREATE '$DB_NAME'. ALL DATA WILL BE LOST!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        pz_log "Aborted"
        return 0
    fi

    export PGPASSWORD="$DB_PASS"
    pz_log "Dropping database '$DB_NAME'..."
    dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" 2>/dev/null || \
    docker exec piezo-ai-db psql -U piezo -c "DROP DATABASE IF EXISTS piezo_ai;" 2>/dev/null || {
        pz_err "Failed to drop database"
        unset PGPASSWORD
        return 1
    }
    unset PGPASSWORD

    pz_db_create || return 1

    pz_log "Running migrations on fresh database..."
    source "$_PZ_VENV_DIR/bin/activate" || return 1
    local sync_url
    sync_url=$(echo "$DB_URL" | sed 's|postgresql+asyncpg://|postgresql://|' | sed 's|postgresql+psycopg://|postgresql://|')
    DATABASE_URL="$sync_url" alembic -c "$ROOT_DIR/packages/db/alembic.ini" upgrade head || {
        pz_err "Migrations failed after reset"
        return 1
    }
    pz_success "Database hard reset complete"
    return 0
}

# ── Stop Docker DB container ─────────────────
pz_db_stop_docker() {
    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        pz_log "Stopping Docker PostgreSQL container..."
        docker compose -f "$ROOT_DIR/docker/docker-compose.yml" stop 2>/dev/null || true
        docker-compose -f "$ROOT_DIR/docker/docker-compose.yml" stop 2>/dev/null || true
        pz_success "Docker PostgreSQL container stopped"
    fi
}

# ── Interactive DB setup mode ────────────────
pz_db_setup_interactive() {
    pz_get_db_vars
    echo ""
    echo -e "${CYAN}--- Database Configuration ---${NC}"
    echo "  1) Docker Container (recommended)"
    echo "  2) Local PostgreSQL installation"
    read -t 20 -p "Select option [1/2] (auto-defaults to 1 in 20s): " db_option || true
    echo ""
    db_option="${db_option:-1}"

    if [[ "$db_option" == "2" ]]; then
        pz_log "Using local PostgreSQL"
        if ! pz_db_is_ready; then
            pz_err "Cannot connect to local PostgreSQL. Ensure it is running."
            return 1
        fi
    else
        pz_log "Using Docker PostgreSQL"
        if ! command -v docker &>/dev/null; then
            pz_err "Docker not installed. Install: https://docker.com/products/docker-desktop"
            return 1
        fi
        if ! docker info &>/dev/null 2>&1; then
            pz_err "Docker daemon is not running. Start Docker Desktop first."
            return 1
        fi
        pz_db_start_docker || return 1

        pz_log "Waiting for database..."
        for i in {1..30}; do
            if pz_db_is_ready; then
                pz_success "Database is ready!"
                return 0
            fi
            echo -n "."
            sleep 1
        done
        echo ""
        pz_warn "Timed out waiting for DB (30s). It may still be starting."
    fi
    return 0
}
