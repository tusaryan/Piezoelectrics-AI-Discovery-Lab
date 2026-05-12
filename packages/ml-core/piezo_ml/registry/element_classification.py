"""
Piezo.AI — Element Classification Registry
==========================================
All element categories, perovskite site assignments, and
element-group definitions. Single source of truth used
by element_registry.py, feature_engineer.py, and all ML modules.

Usage:
    from piezo_ml.registry.element_classification import (
        A_SITE_ELEMENTS, B_SITE_ELEMENTS, DOPANT_ELEMENTS,
        RARE_EARTH_ELEMENTS, ANION_ELEMENTS,
        get_element_category, get_perovskite_site,
        ELEMENT_CATEGORIES, get_category_elements,
    )
"""

from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Element Categories — Single Source of Truth
# ---------------------------------------------------------------------------

class ElementCategory(TypedDict):
    """Category definition with description and examples."""
    name: str
    description: str
    elements: frozenset[str]


ELEMENT_CATEGORIES: dict[str, ElementCategory] = {
    "A_SITE": {
        "name": "A-site cations",
        "description": "Perovskite A-site (12-coordination): alkali, alkaline earth, bismuth, lead",
        "elements": frozenset({"K", "Na", "Li", "Ba", "Ca", "Sr", "Ag", "Bi", "Pb"}),
    },
    "B_SITE": {
        "name": "B-site cations",
        "description": "Perovskite B-site (6-coordination): transition metals, Nb/Ta/Ti/Zr",
        "elements": frozenset({"Nb", "Ta", "Ti", "Zr", "Hf", "Sb", "W", "Mo", "Sn", "Sc", "Fe", "Cr", "Ni", "Co", "In"}),
    },
    "DOPANT": {
        "name": "Dopant / modifier elements",
        "description": "Small amounts for property tuning: acceptor/donor dopants, sintering aids",
        "elements": frozenset({"Cu", "Mn", "Al", "Mg", "Zn", "C", "N"}),
    },
    "RARE_EARTH": {
        "name": "Rare earth elements",
        "description": "Lanthanides used as A-site dopants for TC engineering and property optimization",
        "elements": frozenset({"La", "Nd", "Pr", "Sm", "Eu", "Gd", "Ho"}),
    },
    "ANION": {
        "name": "Anion",
        "description": "Oxygen — the universal anion in oxide piezoelectrics",
        "elements": frozenset({"O"}),
    },
    "TRANSITION_METAL_B": {
        "name": "Transition metal B-site",
        "description": "4d/5d transition metals for high polarizability B-sites",
        "elements": frozenset({"Nb", "Ta", "W", "Mo"}),
    },
}

# Convenience sets for fast lookup
A_SITE_ELEMENTS: frozenset[str] = ELEMENT_CATEGORIES["A_SITE"]["elements"]
B_SITE_ELEMENTS: frozenset[str] = ELEMENT_CATEGORIES["B_SITE"]["elements"]
DOPANT_ELEMENTS: frozenset[str] = ELEMENT_CATEGORIES["DOPANT"]["elements"]
RARE_EARTH_ELEMENTS: frozenset[str] = ELEMENT_CATEGORIES["RARE_EARTH"]["elements"]
ANION_ELEMENTS: frozenset[str] = ELEMENT_CATEGORIES["ANION"]["elements"]
TRANSITION_METAL_B_ELEMENTS: frozenset[str] = ELEMENT_CATEGORIES["TRANSITION_METAL_B"]["elements"]

# ---------------------------------------------------------------------------
# Perovskite Site Mapping
# ---------------------------------------------------------------------------

PEROVSKITE_SITES: dict[str, str] = {
    # A-site (12-coordination cuboctahedral)
    "K": "A", "Na": "A", "Li": "A", "Ba": "A", "Ca": "A", "Sr": "A", "Ag": "A", "Bi": "A", "Pb": "A",
    # B-site (6-coordination octahedral) — including transition metals Cr/Ni/Co and post-transition In
    "Nb": "B", "Ta": "B", "Ti": "B", "Zr": "B", "Hf": "B", "Sb": "B",
    "W": "B", "Mo": "B", "Sn": "B", "Sc": "B", "Fe": "B",
    "Cr": "B", "Ni": "B", "Co": "B", "In": "B",
    # Anion (4-coordination)
    "O": "O",
}


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_element_category(symbol: str) -> str | None:
    """Return the category name for an element."""
    for cat_key, cat_def in ELEMENT_CATEGORIES.items():
        if symbol in cat_def["elements"]:
            return cat_key
    return None


def get_perovskite_site(symbol: str) -> str:
    """Return the perovskite site for an element (A, B, O, dopant)."""
    if symbol in {"K", "Na", "Li", "Ba", "Ca", "Sr", "Ag", "Bi", "Pb"}:
        return "A"
    if symbol in {"Nb", "Ta", "Ti", "Zr", "Hf", "Sb", "W", "Mo", "Sn", "Sc", "Fe", "Cr", "Ni", "Co", "In"}:
        return "B"
    if symbol == "O":
        return "O"
    return "dopant"


def get_category_elements(category: str) -> frozenset[str]:
    """Return all elements in a given category."""
    return ELEMENT_CATEGORIES.get(category, {}).get("elements", frozenset())


def is_a_site(symbol: str) -> bool:
    return symbol in A_SITE_ELEMENTS


def is_b_site(symbol: str) -> bool:
    return symbol in B_SITE_ELEMENTS


def is_dopant(symbol: str) -> bool:
    return symbol in DOPANT_ELEMENTS


def is_rare_earth(symbol: str) -> bool:
    return symbol in RARE_EARTH_ELEMENTS


def is_anion(symbol: str) -> bool:
    return symbol in ANION_ELEMENTS


def is_transition_metal_b(symbol: str) -> bool:
    return symbol in TRANSITION_METAL_B_ELEMENTS


# ---------------------------------------------------------------------------
# All supported elements (union of all categories)
# ---------------------------------------------------------------------------

def get_all_supported_elements() -> frozenset[str]:
    """Union of all elements in all categories."""
    result = set()
    for cat_def in ELEMENT_CATEGORIES.values():
        result |= cat_def["elements"]
    return frozenset(result)


# ---------------------------------------------------------------------------
# Coordination numbers for perovskite structure
# ---------------------------------------------------------------------------

COORDINATION_NUMBERS: dict[str, int] = {
    # A-site: 12-fold cuboctahedral
    **{s: 12 for s in A_SITE_ELEMENTS},
    # B-site: 6-fold octahedral
    **{s: 6 for s in B_SITE_ELEMENTS},
    # Anion: 2-fold (binds to 2 B-site cations)
    **{s: 2 for s in ANION_ELEMENTS},
    # Dopants / non-perovskite: default 6
    **{s: 6 for s in DOPANT_ELEMENTS},
    **{s: 6 for s in RARE_EARTH_ELEMENTS},
}