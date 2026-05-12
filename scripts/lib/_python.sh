#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Python / venv
# ============================================

# ── Constants ────────────────────────────────
_PZ_VENV_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/.venv"
_PZ_PYTHON_CANDIDATES="python3.13 python3.12 python3.11"
_PZ_PYTHON_MINOR_MIN=11
_PZ_PYTHON_MINOR_MAX=13

# ── Locate a compatible Python interpreter ──
# Sets _PZ_PYTHON_CMD and _PZ_PYTHON_VERSION on success. Returns 0 on success.
pz_find_python() {
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
