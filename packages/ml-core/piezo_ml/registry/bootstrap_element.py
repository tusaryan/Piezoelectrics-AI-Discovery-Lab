#!/usr/bin/env python3
"""
Piezo.AI — Element Registry Bootstrap Script
=============================================
Adds new elements to the registry by fetching their properties
from mendeleev and pymatgen, then writes them to the JSON data file.

Usage:
    python -m piezo_ml.registry.bootstrap_element --add Si,Ge,V
    python -m piezo_ml.registry.bootstrap_element --list-pending
    python -m piezo_ml.registry.bootstrap_element --rebuild-all

The bootstrapper is also automatically called on module import when
PENDING_ELEMENTS are detected (via element_registry.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure packages/ml-core is on path
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from piezo_ml.registry.element_classification import (
    ELEMENT_CATEGORIES,
    PEROVSKITE_SITES,
    COORDINATION_NUMBERS,
    get_perovskite_site,
    get_element_category,
    get_all_supported_elements,
)


# ---------------------------------------------------------------------------
# Registry data path (same directory as this script)
# ---------------------------------------------------------------------------

REGISTRY_JSON = Path(__file__).with_name("element_registry_data.json")


# ---------------------------------------------------------------------------
# Source imports
# ---------------------------------------------------------------------------

try:
    from pymatgen.core import Element as PymatgenElement
except ImportError:
    PymatgenElement = None

try:
    from mendeleev import element as mendeleev_element
except ImportError:
    mendeleev_element = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_attr(obj, name: str):
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _extract_valence(pmg) -> int | None:
    if pmg is None:
        return None
    try:
        valence = pmg.valence
        if valence is None:
            return None
        if isinstance(valence, tuple):
            return int(valence[0]) if valence else None
        return int(valence)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Property fetching
# ---------------------------------------------------------------------------

def fetch_element_properties(symbol: str) -> dict:
    """Fetch all properties for a single element from mendeleev/pymatgen."""
    pmg = PymatgenElement(symbol) if PymatgenElement else None
    mend = mendeleev_element(symbol) if mendeleev_element else None

    # Oxidation states
    oxidation_states: list[int] = []
    m_ox = _read_attr(mend, "oxidation_states") if mend else None
    if m_ox:
        oxidation_states = sorted(set(int(x) for x in m_ox))
    elif pmg and pmg.common_oxidation_states:
        oxidation_states = sorted(set(int(x) for x in pmg.common_oxidation_states))

    # Ionization energy
    ionization_energy_ev: float | None = None
    if mend:
        ionenergies = _read_attr(mend, "ionenergies")
        if ionenergies:
            ionization_energy_ev = _safe_float(ionenergies.get(1))

    # Build data dict
    data = {
        "symbol": symbol,
        "atomic_number": int(pmg.Z) if pmg else None,
        "atomic_mass": _safe_float(pmg.atomic_mass) if pmg else None,
        "en_pauling": _safe_float(pmg.X) if pmg else None,
        "atomic_radius_pm": _safe_float(_read_attr(mend, "atomic_radius")),
        "ionic_radius_pm": _safe_float(getattr(pmg, "average_ionic_radius", None)) if pmg else None,
        "covalent_radius_pm": (
            _safe_float(_read_attr(mend, "covalent_radius_pyykko")) or
            _safe_float(getattr(pmg, "atomic_radius_calculated", None))
        ),
        "vdw_radius_pm": _safe_float(_read_attr(mend, "vdw_radius")),
        "melting_point_k": _safe_float(_read_attr(mend, "melting_point")),
        "boiling_point_k": _safe_float(_read_attr(mend, "boiling_point")),
        "electron_affinity_ev": _safe_float(_read_attr(mend, "electron_affinity")),
        "ionization_energy_ev": ionization_energy_ev,
        "valence_electrons": _extract_valence(pmg),
        "group": int(pmg.group) if pmg and pmg.group is not None else None,
        "period": int(pmg.row) if pmg else None,
        "block": str(pmg.block) if pmg else None,
        "density_g_cm3": _safe_float(getattr(pmg, "density_of_solid", None)) if pmg else None,
        "specific_heat_j_gk": _safe_float(_read_attr(mend, "specific_heat_capacity")),
        "thermal_conductivity_w_mk": _safe_float(_read_attr(mend, "thermal_conductivity")),
        "bulk_modulus_gpa": _safe_float(_read_attr(mend, "bulk_modulus")),
        "shear_modulus_gpa": _safe_float(_read_attr(mend, "shear_modulus")),
        "youngs_modulus_gpa": _safe_float(_read_attr(mend, "youngs_modulus")),
        "poisson_ratio": _safe_float(_read_attr(mend, "poissons_ratio")),
        "polarizability_a3": _safe_float(_read_attr(mend, "dipole_polarizability")),
        "oxidation_states": oxidation_states,
        "perovskite_site": get_perovskite_site(symbol),
        "coordination_number": COORDINATION_NUMBERS.get(symbol, 6),
        "is_rare_earth": symbol in ELEMENT_CATEGORIES["RARE_EARTH"]["elements"],
    }
    return data


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, dict]:
    if not REGISTRY_JSON.exists():
        return {}
    with REGISTRY_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict[str, dict]) -> None:
    with REGISTRY_JSON.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    print(f"[bootstrap] Saved {len(registry)} elements to {REGISTRY_JSON}")


# ---------------------------------------------------------------------------
# Core bootstrap logic
# ---------------------------------------------------------------------------

def bootstrap_elements(symbols: list[str]) -> dict[str, dict]:
    """Fetch and persist properties for given element symbols."""
    registry = load_registry()
    added = {}
    errors = []

    for symbol in symbols:
        symbol = symbol.strip()
        if not symbol:
            continue
        try:
            data = fetch_element_properties(symbol)
            registry[symbol] = data
            added[symbol] = data
            print(f"  + {symbol}: site={data['perovskite_site']}, mass={data['atomic_mass']}")
        except Exception as exc:
            errors.append((symbol, str(exc)))
            print(f"  ! {symbol}: {exc}")

    if added:
        save_registry(registry)

    return added


def list_pending() -> list[str]:
    """List elements in PENDING but not yet in registry."""
    from piezo_ml.registry.element_registry import PENDING_ELEMENTS
    registry = load_registry()
    pending_not_bootstrap = [s for s in PENDING_ELEMENTS if s not in registry]
    print(f"PENDING_ELEMENTS: {PENDING_ELEMENTS}")
    print(f"Registry has: {list(registry.keys())}")
    if pending_not_bootstrap:
        print(f"  → {len(pending_not_bootstrap)} pending elements not yet in JSON: {pending_not_bootstrap}")
    else:
        print("  → All pending elements are in the registry JSON.")
    return pending_not_bootstrap


def rebuild_all() -> None:
    """Re-fetch properties for all elements currently in the registry."""
    registry = load_registry()
    if not registry:
        print("[bootstrap] Registry is empty — fetching all from classification...")
        for symbol in sorted(get_all_supported_elements()):
            registry[symbol] = fetch_element_properties(symbol)
    else:
        for symbol in list(registry.keys()):
            registry[symbol] = fetch_element_properties(symbol)
    save_registry(registry)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Piezo.AI Element Registry Bootstrapper")
    parser.add_argument("--add", help="Comma-separated element symbols to add, e.g. Si,Ge,V")
    parser.add_argument("--list-pending", action="store_true", help="List elements waiting to be bootstrapped")
    parser.add_argument("--rebuild-all", action="store_true", help="Re-fetch properties for all elements in registry")
    args = parser.parse_args()

    if args.list_pending:
        list_pending()
    elif args.rebuild_all:
        print("[bootstrap] Rebuilding all elements from sources...")
        rebuild_all()
    elif args.add:
        symbols = [s.strip() for s in args.add.split(",") if s.strip()]
        print(f"[bootstrap] Adding {len(symbols)} element(s): {symbols}")
        added = bootstrap_elements(symbols)
        print(f"[bootstrap] Done — {len(added)} element(s) added.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()