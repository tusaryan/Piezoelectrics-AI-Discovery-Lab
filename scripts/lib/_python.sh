#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Python / venv / pyenv
# ============================================

# ── Constants ────────────────────────────────
_PZ_VENV_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/.venv"
_PZ_PYTHON_CANDIDATES="python3.13 python3.12 python3.11"
_PZ_PYTHON_MINOR_MIN=11
_PZ_PYTHON_MINOR_MAX=13
_PZ_PYTHON_VERSION="3.13.5"

# ── pyenv installation helper ────────────────
pz_install_pyenv() {
    echo ""
    pz_log "pyenv not found. Installing pyenv locally..."
    echo ""

    # Check for Homebrew
    if command -v brew &>/dev/null; then
        pz_log "Installing pyenv via Homebrew..."
        if brew install pyenv; then
            pz_success "pyenv installed via Homebrew"
        else
            pz_err "Failed to install pyenv via Homebrew"
            return 1
        fi
    else
        # Manual install (download and compile Python directly)
        pz_log "Homebrew not found. Will install Python $PZ_PYTHON_VERSION directly..."

        # Check if Xcode command line tools are installed
        if ! xcode-select -p &>/dev/null; then
            pz_warn "Xcode command line tools not installed"
            echo ""
            echo "To install Python locally, you need Xcode command line tools."
            echo "Run: xcode-select --install"
            echo ""
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                return 1
            fi
        fi

        # Try to find system python3.13 or use pyenv
        pz_log "Attempting to install Python $PZ_PYTHON_VERSION..."
        if command -v pyenv &>/dev/null; then
            eval "$(pyenv init -)"
            pyenv install "$_PZ_PYTHON_VERSION" 2>/dev/null && {
                pz_success "Python $_PZ_PYTHON_VERSION installed via pyenv"
                return 0
            }
        fi

        # Last resort: check if there's a compatible Python already
        pz_warn "Could not install Python automatically."
        pz_warn "Please install Python 3.11-3.13 manually or install Homebrew:"
        pz_warn "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        return 1
    fi

    # Initialize pyenv in current shell
    if [ -f "$HOME/.pyenv/bin/pyenv" ]; then
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi

    return 0
}

# ── Setup Python via pyenv (local to project) ─
pz_setup_python_local() {
    local choice="${_PZ_INSTALL_PREF:-ACCEPT_ALL}"

    local has_global=false
    local global_python_cmd=""
    local global_python_ver=""
    
    # Temporarily check system/global Python
    if pz_find_python >/dev/null 2>&1; then
        has_global=true
        global_python_cmd="$_PZ_PYTHON_CMD"
        global_python_ver="$_PZ_PYTHON_VERSION"
    fi

    if [ "$has_global" = true ]; then
        if [ "$choice" = "INDIVIDUAL" ]; then
            pz_info "Detected compatible global Python: $global_python_ver"
            read -p "  Use this global version? [Y/n] (n = install locally from scratch): " ans
            if [[ "$ans" =~ ^[Nn]$ ]]; then
                choice="REJECT_ALL"
            else
                choice="ACCEPT_ALL"
            fi
            echo ""
        fi

        if [ "$choice" != "REJECT_ALL" ]; then
            _PZ_PYTHON_CMD="$global_python_cmd"
            _PZ_PYTHON_VERSION="$global_python_ver"
            pz_success "Using global Python: $_PZ_PYTHON_VERSION"
            return 0
        fi
    fi

    pz_log "Proceeding with local Python installation (isolated pyenv)..."

    # For local isolation, force pyenv to be in project
    export PYENV_ROOT="$ROOT_DIR/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    
    if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
        pz_log "Installing pyenv locally to project..."
        if ! git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT" >/dev/null 2>&1; then
            pz_err "Failed to clone pyenv to $PYENV_ROOT"
            return 1
        fi
    fi
    eval "$(pyenv init -)" 2>/dev/null || true

    if ! pyenv versions 2>/dev/null | grep -q "$_PZ_PYTHON_VERSION"; then
        pz_log "Installing Python $_PZ_PYTHON_VERSION locally via pyenv (this may take a few minutes)..."
        if ! pyenv install "$_PZ_PYTHON_VERSION" >/dev/null; then
            pz_err "Local Python installation failed."
            return 1
        fi
    fi

    cd "$ROOT_DIR"
    pyenv local "$_PZ_PYTHON_VERSION" >/dev/null 2>&1 || true
    cd - >/dev/null

    _PZ_PYTHON_CMD="$PYENV_ROOT/versions/$_PZ_PYTHON_VERSION/bin/python"
    if [ -x "$_PZ_PYTHON_CMD" ]; then
        _PZ_PYTHON_VERSION=$("$_PZ_PYTHON_CMD" --version 2>&1)
        pz_success "Python $_PZ_PYTHON_VERSION configured locally"
        return 0
    fi

    pz_err "Local Python setup failed."
    return 1
}

# ── Locate a compatible Python interpreter ──
# Sets _PZ_PYTHON_CMD and _PZ_PYTHON_VERSION on success. Returns 0 on success.
pz_find_python() {
    # Initialize pyenv if available
    if command -v pyenv &>/dev/null; then
        eval "$(pyenv init -)"
    fi

    # Try PATH candidates first
    for candidate in $_PZ_PYTHON_CANDIDATES; do
        local path
        path=$(command -v "$candidate" 2>/dev/null) || continue
        local minor
        minor=$( "$path" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$minor" -ge "$_PZ_PYTHON_MINOR_MIN" ] && [ "$minor" -le "$_PZ_PYTHON_MINOR_MAX" ]; then
            _PZ_PYTHON_CMD="$path"
            _PZ_PYTHON_VERSION=$("$path" --version 2>&1)
            return 0
        fi
    done

    # Try pyenv versions
    if command -v pyenv &>/dev/null; then
        local pyenv_python
        pyenv_python=$(pyenv which python3 2>/dev/null)
        if [ -n "$pyenv_python" ]; then
            local minor
            minor=$("$pyenv_python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
            if [ "$minor" -ge "$_PZ_PYTHON_MINOR_MIN" ] && [ "$minor" -le "$_PZ_PYTHON_MINOR_MAX" ]; then
                _PZ_PYTHON_CMD="$pyenv_python"
                _PZ_PYTHON_VERSION=$("$pyenv_python" --version 2>&1)
                return 0
            fi
        fi
    fi

    # Try known homebrew /usr/local paths
    for ver in 13 12 11; do
        for prefix in /opt/homebrew/bin /usr/local/bin; do
            if [ -x "${prefix}/python${ver}" ]; then
                _PZ_PYTHON_CMD="${prefix}/python${ver}"
                _PZ_PYTHON_VERSION=$("$_PZ_PYTHON_CMD" --version 2>&1)
                return 0
            fi
        done
    done

    # Fallback: python3 if it happens to be compatible
    if command -v python3 &>/dev/null; then
        local minor
        minor=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$minor" -ge "$_PZ_PYTHON_MINOR_MIN" ] && [ "$minor" -le "$_PZ_PYTHON_MINOR_MAX" ]; then
            _PZ_PYTHON_CMD="python3"
            _PZ_PYTHON_VERSION=$(python3 --version 2>&1)
            return 0
        fi
    fi

    return 1
}

# ── Ensure venv exists with compatible Python ──
# Creates or re-creates the venv if the Python version is incompatible.
pz_ensure_venv() {
    # Ensure we have a valid Python command
    if [ -z "${_PZ_PYTHON_CMD:-}" ]; then
        pz_setup_python_local || return 1
    fi

    if [ ! -f "$_PZ_VENV_DIR/bin/activate" ]; then
        pz_log "Creating virtual environment at $_PZ_VENV_DIR..."
        "$_PZ_PYTHON_CMD" -m venv "$_PZ_VENV_DIR" || return 1
        pz_success "Virtual environment created"
        return 0
    fi

    local venv_minor
    venv_minor=$("$_PZ_VENV_DIR/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$venv_minor" -lt "$_PZ_PYTHON_MINOR_MIN" ] || [ "$venv_minor" -gt "$_PZ_PYTHON_MINOR_MAX" ]; then
        if [ "$venv_minor" = "0" ]; then
            pz_warn "Existing .venv is broken or inaccessible. Recreating..."
        else
            pz_warn "Existing .venv uses Python 3.$venv_minor — incompatible. Recreating..."
        fi
        rm -rf "$_PZ_VENV_DIR"
        "$_PZ_PYTHON_CMD" -m venv "$_PZ_VENV_DIR" || return 1
        pz_success "Virtual environment recreated with $_PZ_PYTHON_VERSION"
    else
        pz_success "Virtual environment exists (Python 3.$venv_minor)"
    fi
    return 0
}

# ── Activate venv ────────────────────────────
pz_activate_venv() {
    if [ -f "$_PZ_VENV_DIR/bin/activate" ]; then
        source "$_PZ_VENV_DIR/bin/activate"
        return 0
    fi
    return 1
}

# ── Install a package in editable mode ──────
# $1 = package spec (e.g. "packages/db")
# $2 = optional extras (e.g. "[symbolic]")
# Returns 0 on success, non-zero on failure.
pz_pip_install() {
    local pkg="$1"
    local extras="${2:-}"
    pz_log "  Installing $pkg${extras}..."
    if source "$_PZ_VENV_DIR/bin/activate" && \
       pip install --upgrade pip --quiet 2>/dev/null && \
       pip install -e "${pkg}${extras}" 2>&1; then
        pz_success "$pkg installed"
        return 0
    else
        pz_err "Failed to install $pkg${extras}"
        return 1
    fi
}

# ── Verify critical Python packages are importable ──
# Checks a list of (import_name, pip_package) pairs.
# If any are missing, prompts the user to install them.
# Returns 0 on success, 1 if critical deps are still missing.
pz_check_python_deps() {
    local python_bin="$_PZ_VENV_DIR/bin/python"
    if [ ! -x "$python_bin" ]; then
        pz_err "Python venv not found. Run: bash scripts/dev.sh setup"
        return 1
    fi

    # Critical packages: "import_name:pip_package" pairs
    local deps=(
        "sklearn:scikit-learn"
        "xgboost:xgboost"
        "lightgbm:lightgbm"
        "optuna:optuna"
        "numpy:numpy"
        "pandas:pandas"
        "joblib:joblib"
        "chemparse:chemparse"
        "pymatgen:pymatgen"
        "mendeleev:mendeleev"
        "shap:shap"
        "matplotlib:matplotlib"
        "reportlab:reportlab"
        "pymoo:pymoo"
        "fastapi:fastapi"
        "uvicorn:uvicorn"
        "sqlalchemy:sqlalchemy"
        "alembic:alembic"
    )

    local missing_imports=()
    local missing_packages=()

    for dep in "${deps[@]}"; do
        local import_name="${dep%%:*}"
        local pip_name="${dep##*:}"
        if ! "$python_bin" -c "import $import_name" 2>/dev/null; then
            missing_imports+=("$import_name")
            missing_packages+=("$pip_name")
        fi
    done

    if [ ${#missing_imports[@]} -eq 0 ]; then
        pz_success "All critical Python dependencies are available"
        return 0
    fi

    echo ""
    pz_warn "Missing Python packages detected (${#missing_imports[@]}):"
    for i in "${!missing_imports[@]}"; do
        echo "  • ${missing_imports[$i]} (pip: ${missing_packages[$i]})"
    done
    echo ""

    read -p "Install missing packages now? (Y/n): " install_choice
    echo ""

    if [[ "$install_choice" =~ ^[Nn]$ ]]; then
        pz_warn "Skipping installation. Some features may not work."
        pz_warn "To install later: source .venv/bin/activate && pip install ${missing_packages[*]}"
        return 1
    fi

    # Install missing packages
    pz_log "Installing missing packages..."
    source "$_PZ_VENV_DIR/bin/activate"

    local install_failed=false
    for pkg in "${missing_packages[@]}"; do
        pz_log "  Installing $pkg..."
        if pip install "$pkg" --quiet 2>&1; then
            pz_success "  $pkg installed"
        else
            pz_err "  Failed to install $pkg"
            install_failed=true
        fi
    done

    if [ "$install_failed" = true ]; then
        pz_warn "Some packages failed to install. Check network/build tools."
        return 1
    fi

    pz_success "All missing packages installed successfully"
    return 0
}