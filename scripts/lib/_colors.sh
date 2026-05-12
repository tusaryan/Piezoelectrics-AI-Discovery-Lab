#!/usr/bin/env bash
# ============================================
# Piezo.AI — Shell Library: Colors & Logging
# ============================================

# ── Colors ──────────────────────────────────
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export CYAN='\033[0;36m'
export BOLD='\033[1m'
export NC='\033[0m'

# ── Logging helpers ──────────────────────────
# These accumulate warnings/errors for end-of-command summary.
_pz_warnings=()
_pz_errors=()

pz_log()     { echo -e "${CYAN}[piezo-ai]${NC} $1"; }
pz_success() { echo -e "${GREEN}[✓]${NC} $1"; }
pz_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
    _pz_warnings+=("$1")
}
pz_err() {
    echo -e "${RED}[✗]${NC} $1"
    _pz_errors+=("$1")
}
pz_info()    { echo -e "  $1"; }

# ── Summary printer ──────────────────────────
# Call once at the end of any setup/start command.
pz_print_summary() {
    local exit_code=$?
    echo ""
    echo "=========================================================================="
    if [ $exit_code -eq 0 ] && [ ${#_pz_errors[@]} -eq 0 ]; then
        if [ ${#_pz_warnings[@]} -eq 0 ]; then
            echo -e "${GREEN}✅ SUCCESS!${NC}"
        else
            echo -e "${YELLOW}⚠️  COMPLETED WITH WARNINGS:${NC}"
            for w in "${_pz_warnings[@]}"; do echo -e "   -> $w"; done
        fi
    else
        echo -e "${RED}❌ FAILED:${NC}"
        for e in "${_pz_errors[@]}"; do echo -e "   -> $e"; done
        if [ ${#_pz_warnings[@]} -gt 0 ]; then
            echo -e "${YELLOW}Warnings:${NC}"
            for w in "${_pz_warnings[@]}"; do echo -e "   -> $w"; done
        fi
    fi
    echo "=========================================================================="
    # Reset accumulators
    _pz_warnings=()
    _pz_errors=()
}
