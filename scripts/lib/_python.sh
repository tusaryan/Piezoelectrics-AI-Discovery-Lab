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
    # Try to find existing pyenv
    if command -v pyenv &>/dev/null; then
        eval "$(pyenv init -)"
    fi

    # Check for pyenv in common locations
    if [ -f "$HOME/.pyenv/bin/pyenv" ]; then
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi

    # Try to use pyenv to install/manage Python
    if command -v pyenv &>/dev/null; then
        pz_log "Using pyenv to manage Python..."

        # Check if Python version is installed
        if ! pyenv versions | grep -q "$_PZ_PYTHON_VERSION"; then
            pz_log "Installing Python $_PZ_PYTHON_VERSION via pyenv..."
            if pyenv install "$_PZ_PYTHON_VERSION"; then
                pz_success "Python $_PZ_PYTHON_VERSION installed"
            else
                pz_err "Failed to install Python $_PZ_PYTHON_VERSION"
                return 1
            fi
        fi

        # Set local version for this project
        cd "$ROOT_DIR"
        pyenv local "$_PZ_PYTHON_VERSION"
        cd - >/dev/null

        # Export for scripts
        _PZ_PYTHON_CMD="$PYENV_ROOT/versions/$_PZ_PYTHON_VERSION/bin/python"
        _PZ_PYTHON_VERSION=$("$_PZ_PYTHON_CMD" --version 2>&1)

        pz_success "Python $_PZ_PYTHON_VERSION configured (via pyenv)"
        return 0
    fi

    # pyenv not found - try to install it
    pz_install_pyenv || return 1

    # Retry after install
    if command -v pyenv &>/dev/null; then
        eval "$(pyenv init -)"
        pz_setup_python_local
        return $?
    fi

    # Fallback: try system Python
    pz_log "pyenv installation failed. Trying system Python..."
    if pz_find_python; then
        pz_success "Using system Python: $_PZ_PYTHON_VERSION"
        return 0
    fi

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
        pz_warn "Existing .venv uses Python 3.$venv_minor — incompatible. Recreating..."
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