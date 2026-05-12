#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Network Diagnostics
# ============================================

# ── Check pypi.org reachability ──────────────
# Returns 0 if reachable, 1 if not.
pz_check_pypi() {
    if curl -s --max-time 5 https://pypi.org/simple/ >/dev/null 2>&1; then
        return 0
    fi
    # Try alternative DNS
    if curl -s --max-time 5 --dns-servers 8.8.8.8 https://pypi.org/simple/ >/dev/null 2>&1; then
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

    # Check internet
    if curl -s --max-time 5 https://pypi.org >/dev/null 2>&1; then
        pz_success "Internet: reachable (pypi.org)"
    else
        pz_warn "Internet: pypi.org is NOT reachable"
        pz_info "Possible causes:"
        pz_info "  - VPN / proxy blocking pypi.org"
        pz_info "  - Firewall / DNS outage"
        pz_info "  - No internet connection"
        pz_info ""
        pz_info "Try:"
        pz_info "  - Disable VPN / proxy temporarily"
        pz_info "  - Check DNS:   nslookup pypi.org"
        pz_info "  - Test:        curl https://pypi.org/simple/"
    fi

    # Check DNS
    if command -v nslookup &>/dev/null; then
        if nslookup pypi.org &>/dev/null; then
            pz_success "DNS: pypi.org resolves"
        else
            pz_warn "DNS: pypi.org does NOT resolve"
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
