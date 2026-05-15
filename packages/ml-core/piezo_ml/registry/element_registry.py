"""Piezo.AI central element registry with auto-bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pymatgen.core import Element

try:
    from mendeleev import element as mendeleev_element
except Exception:  # pragma: no cover - optional import fallback
    mendeleev_element = None

SUPPORTED_ELEMENTS: frozenset[str] = frozenset(
    {
        "K", "Na", "Li", "Ba", "Ca", "Sr", "Ag",
        "Bi", "Pb",
        "Nb", "Ta", "Ti", "Zr", "Hf", "Sb", "W", "Mo", "Sn", "Sc", "Fe",
        "Cu", "Mn", "Al", "Mg", "Zn",
        "La", "Nd", "Pr", "Sm", "Eu", "Gd", "Ho",
        "O",
        # S9.5 additions: new elements for expanded material space
        "H", "B", "N", "C", "Co", "Cr", "In", "Si", "Ni",
    }
)

PENDING_ELEMENTS: list[str] = []
REGISTRY_DATA_PATH = Path(__file__).with_name("element_registry_data.json")

RARE_EARTHS = {"La", "Nd", "Pr", "Sm", "Eu", "Gd", "Ho"}
A_SITE = {"K", "Na", "Li", "Ba", "Ca", "Sr", "Ag", "Bi", "Pb"}
B_SITE = {"Nb", "Ta", "Ti", "Zr", "Hf", "Sb", "W", "Mo", "Sn", "Sc", "Fe", "Cr", "Ni", "Co", "In"}
DOPANTS = {"Cu", "Mn", "Al", "Mg", "Zn", "C", "N", "B", "Si", "Co", "Cr", "Ni", "In"}

PROPERTY_KEYS: tuple[str, ...] = (
    "atomic_number",
    "symbol",
    "atomic_mass",
    "en_pauling",
    "atomic_radius_pm",
    "ionic_radius_pm",
    "covalent_radius_pm",
    "vdw_radius_pm",
    "melting_point_k",
    "boiling_point_k",
    "electron_affinity_ev",
    "ionization_energy_ev",
    "valence_electrons",
    "group",
    "period",
    "block",
    "density_g_cm3",
    "specific_heat_j_gk",
    "thermal_conductivity_w_mk",
    "bulk_modulus_gpa",
    "shear_modulus_gpa",
    "youngs_modulus_gpa",
    "poisson_ratio",
    "polarizability_a3",
    "oxidation_states",
    "coordination_number",
    "perovskite_site",
    "is_rare_earth",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def _detect_perovskite_site(symbol: str) -> str:
    if symbol == "O":
        return "O"
    if symbol in A_SITE:
        return "A"
    if symbol in B_SITE:
        return "B"
    if symbol in DOPANTS or symbol in RARE_EARTHS:
        return "dopant"
    return "dopant"


def _extract_valence_electrons(pmg: Element) -> int | None:
    try:
        valence = pmg.valence
    except Exception:
        valence = None
    if valence is None:
        return None
    if isinstance(valence, tuple):
        if len(valence) == 2:
            return int(valence[0])
        return int(valence[0]) if valence else None
    return int(valence)


def _estimate_coordination_number(symbol: str) -> int:
    site = _detect_perovskite_site(symbol)
    if site == "A":
        return 12
    if site == "B":
        return 6
    if site == "O":
        return 2
    return 6


def _build_from_sources(symbol: str) -> dict[str, Any]:
    pmg = Element(symbol)
    mendeleev_obj = mendeleev_element(symbol) if mendeleev_element else None

    oxidation_states = []
    m_oxidation_states = _read_attr(mendeleev_obj, "oxidation_states") if mendeleev_obj else None
    if m_oxidation_states:
        oxidation_states = sorted(set(int(x) for x in m_oxidation_states))
    elif pmg.common_oxidation_states:
        oxidation_states = sorted(set(int(x) for x in pmg.common_oxidation_states))

    ionization_energy_ev = None
    ionenergies = _read_attr(mendeleev_obj, "ionenergies") if mendeleev_obj else None
    if ionenergies:
        ionization_energy_ev = _safe_float(ionenergies.get(1))

    data = {
        "atomic_number": int(pmg.Z),
        "symbol": symbol,
        "atomic_mass": _safe_float(pmg.atomic_mass),
        "en_pauling": _safe_float(pmg.X),
        "atomic_radius_pm": _safe_float(_read_attr(mendeleev_obj, "atomic_radius")) or _safe_float(getattr(pmg, "atomic_radius", None)),
        "ionic_radius_pm": _safe_float(getattr(pmg, "average_ionic_radius", None)),
        "covalent_radius_pm": _safe_float(_read_attr(mendeleev_obj, "covalent_radius_pyykko")) or _safe_float(getattr(pmg, "atomic_radius_calculated", None)),
        "vdw_radius_pm": _safe_float(_read_attr(mendeleev_obj, "vdw_radius")),
        "melting_point_k": _safe_float(_read_attr(mendeleev_obj, "melting_point")),
        "boiling_point_k": _safe_float(_read_attr(mendeleev_obj, "boiling_point")),
        "electron_affinity_ev": _safe_float(_read_attr(mendeleev_obj, "electron_affinity")),
        "ionization_energy_ev": ionization_energy_ev,
        "valence_electrons": _extract_valence_electrons(pmg),
        "group": int(pmg.group) if pmg.group is not None else None,
        "period": int(pmg.row),
        "block": str(pmg.block),
        "density_g_cm3": _safe_float(getattr(pmg, "density_of_solid", None)),
        "specific_heat_j_gk": _safe_float(_read_attr(mendeleev_obj, "specific_heat_capacity")),
        "thermal_conductivity_w_mk": _safe_float(_read_attr(mendeleev_obj, "thermal_conductivity")),
        "bulk_modulus_gpa": _safe_float(_read_attr(mendeleev_obj, "bulk_modulus")),
        "shear_modulus_gpa": _safe_float(_read_attr(mendeleev_obj, "shear_modulus")),
        "youngs_modulus_gpa": _safe_float(_read_attr(mendeleev_obj, "youngs_modulus")),
        "poisson_ratio": _safe_float(_read_attr(mendeleev_obj, "poissons_ratio")),
        "polarizability_a3": _safe_float(_read_attr(mendeleev_obj, "dipole_polarizability")),
        "oxidation_states": oxidation_states,
        "coordination_number": _estimate_coordination_number(symbol),
        "perovskite_site": _detect_perovskite_site(symbol),
        "is_rare_earth": symbol in RARE_EARTHS,
    }
    return data


def _load_registry_file() -> dict[str, dict[str, Any]]:
    if not REGISTRY_DATA_PATH.exists():
        return {}
    with REGISTRY_DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def bootstrap_pending_elements() -> dict[str, dict[str, Any]]:
    registry = _load_registry_file()
    to_bootstrap = sorted(set(PENDING_ELEMENTS) - set(registry.keys()))
    if not to_bootstrap:
        return {}

    for symbol in to_bootstrap:
        registry[symbol] = _build_from_sources(symbol)
    return {symbol: registry[symbol] for symbol in to_bootstrap}


def _load_or_bootstrap_registry() -> dict[str, dict[str, Any]]:
    pending_registry = bootstrap_pending_elements()
    registry = _load_registry_file()
    registry.update(pending_registry)

    # Include user-added elements from customizations file (T3: persistence fix)
    custom_path = Path(__file__).resolve().parents[3] / "resources" / ".settings-customizations.json"
    user_added: set[str] = set()
    if custom_path.exists():
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                customs = json.load(f)
                user_added = set(customs.get("added_elements", []))
        except Exception:
            pass

    # All elements to load = default SUPPORTED + user-added
    all_elements = SUPPORTED_ELEMENTS | user_added
    missing = all_elements - set(registry.keys())
    if missing:
        for symbol in sorted(missing):
            try:
                registry[symbol] = _build_from_sources(symbol)
            except Exception:
                pass  # skip elements that can't be bootstrapped
    return {symbol: registry[symbol] for symbol in sorted(all_elements) if symbol in registry}


ELEMENT_REGISTRY: dict[str, dict[str, Any]] = _load_or_bootstrap_registry()


def is_element_supported(symbol: str) -> bool:
    return symbol in ELEMENT_REGISTRY


def get_unsupported_elements(elements: set[str]) -> set[str]:
    return {x for x in elements if x not in ELEMENT_REGISTRY}


def get_supported_elements_list() -> list[str]:
    return sorted(ELEMENT_REGISTRY.keys())


def get_element_count() -> int:
    return len(ELEMENT_REGISTRY)


def get_element_properties(symbol: str) -> dict[str, Any]:
    if symbol not in ELEMENT_REGISTRY:
        raise KeyError(f"Unsupported element: {symbol}")
    return ELEMENT_REGISTRY[symbol]
