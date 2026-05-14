#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Network Diagnostics
# ============================================

# ── Check pypi.org reachability via curl ──────
# Returns 0 if reachable, 1 if not.
pz_check_pypi_curl() {
    # Use -I (HEAD) to prevent downloading the massive 30MB+ simple index, which causes timeouts
    if curl -sI --max-time 5 https://pypi.org/simple/ >/dev/null 2>&1; then
        return 0
    fi
    # Try alternative DNS
    if curl -sI --max-time 5 --dns-servers 8.8.8.8 https://pypi.org/simple/ >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ── Check pypi.org reachability via pip itself ─
# Returns 0 if pip can actually download, 1 if not.
# This is the authoritative check — curl and pip can differ due to
# proxy env vars, SSL config, or pip's own DNS caching.
pz_check_pypi_pip() {
    if [ ! -f "$_PZ_VENV_DIR/bin/pip" ]; then
        return 1  # pip not installed yet — can't test
    fi
    if "$_PZ_VENV_DIR/bin/pip" download --no-deps --dest /tmp/pz_net_test \
        --index-url https://pypi.org/simple/ setuptools &>/dev/null; then
        rm -rf /tmp/pz_net_test 2>/dev/null
        return 0
    fi
    rm -rf /tmp/pz_net_test 2>/dev/null
    return 1
}

# ── Check pypi.org reachability (authoritative: uses pip when available) ──
# Falls back to curl if pip isn't installed yet (pre-setup check).
pz_check_pypi() {
    if [ -f "$_PZ_VENV_DIR/bin/pip" ]; then
        pz_check_pypi_pip
    else
        pz_check_pypi_curl
    fi
}

# ── Diagnose network issues ───────────────────
pz_diagnose_network() {
    pz_log "Running network diagnostics..."

    # ── curl reachability (quick check) ──
    pz_log "  Checking internet via curl..."
    if pz_check_pypi_curl; then
        pz_success "  [curl] pypi.org is reachable"
    else
        pz_warn "  [curl] pypi.org is NOT reachable via curl"
        pz_info "  Possible causes:"
        pz_info "    - VPN / proxy blocking pypi.org"
        pz_info "    - Firewall / DNS outage"
        pz_info "    - No internet connection"
    fi

    # ── pip reachability (authoritative check) ──
    pz_log "  Checking internet via pip..."
    if [ -f "$_PZ_VENV_DIR/bin/pip" ]; then
        if "$_PZ_VENV_DIR/bin/pip" download --no-deps --dest /tmp/pz_pip_test \
            --index-url https://pypi.org/simple/ setuptools &>/dev/null; then
            pz_success "  [pip] pypi.org is reachable"
            rm -rf /tmp/pz_pip_test 2>/dev/null
        else
            pz_warn "  [pip] pypi.org is NOT reachable via pip (network issue or broken venv)"
            pz_info "  Try:"
            pz_info "    - Disable VPN / proxy"
            pz_info "    - Check pip proxy: pip config list"
            pz_info "    - Check pip SSL: pip config list --format freeze | grep cert"
            pz_info "    - Force DNS: pip install --trusted-host pypi.org ..."
            rm -rf /tmp/pz_pip_test 2>/dev/null
        fi
    else
        pz_info "  [pip] not installed yet — skipping pip-level check"
    fi

    # ── pip cache status ──
    local cache_dir
    cache_dir=$(pip cache dir 2>/dev/null)
    if [ -n "$cache_dir" ] && [ -d "$cache_dir" ]; then
        local cached_pkgs
        cached_pkgs=$(find "$cache_dir" -type f -name "*.whl" 2>/dev/null | wc -l | tr -d ' ')
        pz_info "  pip cache dir: $cache_dir"
        pz_info "    ($cached_pkgs cached package files — used only when package version matches)"
    fi

    # ── DNS check ──
    if command -v nslookup &>/dev/null; then
        if nslookup pypi.org &>/dev/null; then
            pz_success "  DNS: pypi.org resolves"
        else
            pz_warn "  DNS: pypi.org does NOT resolve"
        fi
    fi

    # Check pip cache
    local cache_dir
    cache_dir=$(source "$_PZ_VENV_DIR/bin/activate" 2>/dev/null && pip cache dir 2>/dev/null)
    if [ -n "$cache_dir" ] && [ -d "$cache_dir" ]; then
        pz_info "pip cache: $cache_dir"
    fi
}

# ── Install pip wheel directly (offline fallback) ──
# Downloads wheel from URL if network is working.
pz_pip_install_url() {
    local pkg_spec="$1"   # e.g. "setuptools>=75.0"
    local wheel_url="$2"  # e.g. "https://files.pythonhosted.org/..."

    if ! pz_check_pypi; then
        pz_err "Cannot reach pypi.org — cannot download $pkg_spec"
        pz_diagnose_network
        return 1
    fi

    source "$_PZ_VENV_DIR/bin/activate" || return 1
    if pip install "$pkg_spec" 2>&1; then
        pz_success "$pkg_spec installed"
        return 0
    fi
    pz_err "pip install failed for $pkg_spec"
    return 1
}
