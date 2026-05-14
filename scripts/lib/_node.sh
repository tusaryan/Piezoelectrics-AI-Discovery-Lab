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
    local choice="${_PZ_INSTALL_PREF:-ACCEPT_ALL}"

    # Check for global node first
    local has_global=false
    local global_node_ver=""
    
    # Check system node
    if command -v node &>/dev/null; then
        local major
        major=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
        if [ "${major:-0}" -ge 20 ]; then
            has_global=true
            global_node_ver=$(node -v)
        fi
    elif [ -s "$HOME/.nvm/nvm.sh" ]; then
        # Check global nvm
        export NVM_DIR="$HOME/.nvm"
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
        if nvm use 20 >/dev/null 2>&1 || nvm use default >/dev/null 2>&1; then
            local major
            major=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
            if [ "${major:-0}" -ge 20 ]; then
                has_global=true
                global_node_ver=$(node -v)
            fi
        fi
    fi

    if [ "$has_global" = true ]; then
        if [ "$choice" = "INDIVIDUAL" ]; then
            pz_info "Detected compatible global Node.js: $global_node_ver"
            read -p "  Use this global version? [Y/n] (n = install locally from scratch): " ans
            if [[ "$ans" =~ ^[Nn]$ ]]; then
                choice="REJECT_ALL"
            else
                choice="ACCEPT_ALL"
            fi
            echo ""
        fi

        if [ "$choice" != "REJECT_ALL" ]; then
            NODE_VERSION="$global_node_ver"
            pz_success "Using global Node.js: $global_node_ver"
            return 0
        fi
    fi

    pz_log "Proceeding with local Node.js installation (nvm)..."

    # Try to load existing local nvm
    if [ -s "$_PZ_NVM_DIR/nvm.sh" ]; then
        export NVM_DIR="$_PZ_NVM_DIR"
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
    fi

    # Check if nvm exists
    if ! command -v nvm &>/dev/null; then
        pz_install_nvm || { pz_err "Failed to install local nvm"; return 1; }
        if [ -s "$_PZ_NVM_DIR/nvm.sh" ]; then
            export NVM_DIR="$_PZ_NVM_DIR"
            source "$NVM_DIR/nvm.sh" 2>/dev/null || true
        fi
    fi

    if command -v nvm &>/dev/null; then
        if ! nvm ls "$_PZ_NODE_VERSION" 2>/dev/null | grep -q "v$_PZ_NODE_VERSION"; then
            pz_log "Installing Node.js $_PZ_NODE_VERSION locally via nvm..."
            nvm install "$_PZ_NODE_VERSION" >/dev/null || return 1
        fi
        
        nvm use "$_PZ_NODE_VERSION" >/dev/null || return 1
        NODE_VERSION=$(node -v)
        echo "$_PZ_NODE_VERSION" > "$ROOT_DIR/.nvmrc"
        pz_success "Node.js $NODE_VERSION configured locally"
        return 0
    fi

    pz_err "Local Node.js installation failed."
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

    local pnpm_ver
    if command -v pnpm &>/dev/null; then
        pnpm_ver=$(pnpm --version 2>/dev/null || echo "ERROR")
        if [ "$pnpm_ver" != "ERROR" ]; then
            pz_success "pnpm $pnpm_ver"
            return 0
        fi
        pz_warn "Existing pnpm is broken or incompatible with Node $NODE_VERSION. Reinstalling..."
    fi

    pz_log "Installing pnpm@9 globally via npm..."
    if npm install -g pnpm@9.15.4 >/dev/null 2>&1; then
        pnpm_ver=$(pnpm --version 2>/dev/null || echo "ERROR")
        if [ "$pnpm_ver" != "ERROR" ]; then
            pz_success "pnpm $pnpm_ver installed successfully"
            return 0
        fi
        pz_warn "pnpm installed but still broken. Your PATH may be prioritizing a broken Homebrew pnpm."
    fi
    
    pz_err "Failed to install a working pnpm. Frontend may not work properly."
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