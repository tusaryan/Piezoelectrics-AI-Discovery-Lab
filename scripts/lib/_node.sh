#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Node.js / nvm
# ============================================

# ── Constants ────────────────────────────────
_PZ_NODE_VERSION="20"
_PZ_NVM_DIR="$ROOT_DIR/.nvm"

# ── Install nvm locally (to project directory) ─
pz_install_nvm() {
    echo ""
    pz_log "nvm not found. Installing nvm locally to project..."

    # Create .nvm directory in project
    mkdir -p "$_PZ_NVM_DIR"

    # Download and install nvm
    local nvm_version="v0.39.7"
    pz_log "Downloading nvm..."

    if curl -o "$_PZ_NVM_DIR/nvm.sh" -sL "https://raw.githubusercontent.com/nvm-sh/nvm/${nvm_version}/nvm.sh"; then
        chmod +x "$_PZ_NVM_DIR/nvm.sh"

        # Add to shell profile
        local profile_file="$HOME/.zshrc"
        if [ -f "$HOME/.bashrc" ]; then
            profile_file="$HOME/.bashrc"
        fi

        # Check if nvm is already in profile
        if ! grep -q "Piezo.AI nvm" "$profile_file" 2>/dev/null; then
            echo "" >> "$profile_file"
            echo "# Piezo.AI nvm setup" >> "$profile_file"
            echo "export PIEZO_NVM_DIR=\"$ROOT_DIR/.nvm\"" >> "$profile_file"
            echo "[ -s \"\$PIEZO_NVM_DIR/nvm.sh\" ] && source \"\$PIEZO_NVM_DIR/nvm.sh\"" >> "$profile_file"
            pz_info "Added nvm to $profile_file"
        fi

        pz_success "nvm installed locally"
        return 0
    else
        pz_err "Failed to download nvm"
        return 1
    fi
}

# ── Setup Node.js via nvm (local to project) ─
pz_setup_node_local() {
    # Try to load existing nvm
    if [ -s "$_PZ_NVM_DIR/nvm.sh" ]; then
        export NVM_DIR="$_PZ_NVM_DIR"
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
    fi

    # Check if nvm exists
    if ! command -v nvm &>/dev/null; then
        # Try to install nvm
        pz_install_nvm || return 1

        # Retry loading
        if [ -s "$_PZ_NVM_DIR/nvm.sh" ]; then
            export NVM_DIR="$_PZ_NVM_DIR"
            source "$NVM_DIR/nvm.sh" 2>/dev/null || true
        fi
    fi

    # Now try to use nvm
    if command -v nvm &>/dev/null; then
        pz_log "Using nvm to manage Node.js..."

        # Check if Node 20 is installed
        if ! nvm ls "$_PZ_NODE_VERSION" 2>/dev/null | grep -q "v$_PZ_NODE_VERSION"; then
            pz_log "Installing Node.js $_PZ_NODE_VERSION via nvm..."
            echo ""
            if nvm install "$_PZ_NODE_VERSION"; then
                pz_success "Node.js $_PZ_NODE_VERSION installed"
            else
                pz_err "Failed to install Node.js $_PZ_NODE_VERSION"
                return 1
            fi
        fi

        # Use Node 20
        if nvm use "$_PZ_NODE_VERSION" 2>/dev/null; then
            NODE_VERSION=$(node -v)
            pz_success "Node.js $NODE_VERSION configured (via nvm)"

            # Create .nvmrc for project
            echo "$_PZ_NODE_VERSION" > "$ROOT_DIR/.nvmrc"

            return 0
        fi
    fi

    # Fallback: try system node
    pz_log "nvm setup failed. Trying system Node.js..."
    if pz_ensure_node; then
        pz_success "Using system Node.js: $NODE_VERSION"
        return 0
    fi

    return 1
}

# ── Ensure Node.js >= 20 is available ───────
# Tries: nvm → system node → fail.
# Sets NODE_VERSION on success.
pz_ensure_node() {
    # Check if we have local nvm
    if [ -s "$_PZ_NVM_DIR/nvm.sh" ]; then
        export NVM_DIR="$_PZ_NVM_DIR"
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
    fi

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
        pz_log "Loading global nvm..."
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

    pz_warn "Node.js 20+ not found."
    pz_warn "Run: bash scripts/dev.sh setup (will install nvm + Node.js locally)"
    return 1
}

# ── Ensure pnpm is installed ─────────────────
pz_ensure_pnpm() {
    # Ensure we have node first
    pz_ensure_node || return 1

    if command -v pnpm &>/dev/null; then
        pz_success "pnpm $(pnpm --version)"
        return 0
    fi
    pz_log "Installing pnpm globally..."
    if npm install -g pnpm 2>/dev/null; then
        pz_success "pnpm installed"
        return 0
    fi
    pz_warn "Failed to install pnpm. Frontend may not work properly."
    return 1
}

# ── Run pnpm install in the project root ─────
pz_pnpm_install() {
    # Ensure node is available
    pz_ensure_node || {
        pz_warn "Node.js not available - skipping pnpm install"
        return 1
    }

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