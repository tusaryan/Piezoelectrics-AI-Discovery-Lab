#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Node.js
# ============================================

# ── Ensure Node.js >= 20 is available ───────
# Tries: system node → nvm load → fail.
# Sets NODE_VERSION on success.
pz_ensure_node() {
    # Check if node is already usable
    if command -v node &>/dev/null; then
        local major
        major=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
        if [ "${major:-0}" -ge 20 ]; then
            NODE_VERSION=$(node -v)
            return 0
        fi
    fi

    # Try to load nvm
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        pz_log "Loading nvm..."
        export NVM_DIR="$HOME/.nvm"
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
        nvm use 20 2>/dev/null || nvm use default 2>/dev/null || true
        if command -v node &>/dev/null; then
            major=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
            if [ "${major:-0}" -ge 20 ]; then
                NODE_VERSION=$(node -v)
                pz_success "Node.js $NODE_VERSION (via nvm)"
                return 0
            fi
        fi
    fi

    pz_err "Node.js 20+ not found."
    pz_err "Fix: nvm install 20 && nvm use 20 && npm i -g pnpm"
    return 1
}

# ── Ensure pnpm is installed ─────────────────
pz_ensure_pnpm() {
    if command -v pnpm &>/dev/null; then
        pz_success "pnpm $(pnpm --version)"
        return 0
    fi
    pz_log "Installing pnpm globally..."
    if npm install -g pnpm 2>/dev/null; then
        pz_success "pnpm installed"
        return 0
    fi
    pz_err "Failed to install pnpm. Run manually: npm i -g pnpm"
    return 1
}

# ── Run pnpm install in the project root ─────
pz_pnpm_install() {
    if ! command -v pnpm &>/dev/null; then
        pz_err "pnpm not available — cannot install frontend deps"
        return 1
    fi
    if [ ! -d "apps/web/node_modules" ] || [ ! -d "node_modules" ]; then
        pz_log "Running pnpm install..."
        if pnpm install 2>&1; then
            pz_success "Frontend dependencies installed"
            return 0
        else
            pz_err "pnpm install failed"
            return 1
        fi
    else
        pz_success "Frontend dependencies already installed"
    fi
    return 0
}
