"""
Piezo.AI — Central Element Registry
=====================================
SINGLE SOURCE OF TRUTH for all supported elements in the platform.

This registry is referenced by:
- Formula parser/validator (validates element symbols)
- Feature engineer (reads physics properties — S3)
- API (returns supported element list)
- Settings UI (shows supported elements, manages PENDING_ELEMENTS)

S2 version: Symbol list only (for formula validation at upload).
S3 will add: Full ELEMENT_REGISTRY dict with 27+ pre-computed physics
             properties per element, auto-bootstrap from mendeleev.

See 01-architecture-and-sections.md §5 for full registry design.

33 elements supported in v2.1:
  A-site (alkali/alkaline): K, Na, Li, Ba, Ca, Sr, Ag
  A-site (bismuth family): Bi, Pb
  B-site (transition metals): Nb, Ta, Ti, Zr, Hf, Sb, W, Mo, Sn, Sc, Fe
  Dopants / modifiers: Cu, Mn, Al, Mg, Zn
  Rare earth dopants: La, Nd, Pr, Sm, Eu, Gd, Ho
  Anion: O
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Supported Elements — the canonical set for v2.1
# ---------------------------------------------------------------------------
SUPPORTED_ELEMENTS: frozenset[str] = frozenset({
    # A-site (alkali / alkaline earth)
    "K", "Na", "Li", "Ba", "Ca", "Sr", "Ag",
    # A-site (bismuth family)
    "Bi", "Pb",
    # B-site (transition metals)
    "Nb", "Ta", "Ti", "Zr", "Hf", "Sb", "W", "Mo", "Sn", "Sc", "Fe",
    # Dopants / modifiers
    "Cu", "Mn", "Al", "Mg", "Zn",
    # Rare earth dopants
    "La", "Nd", "Pr", "Sm", "Eu", "Gd", "Ho",
    # Anion
    "O",
})

# ---------------------------------------------------------------------------
# Pending Elements — awaiting auto-bootstrap in S3
# ---------------------------------------------------------------------------
# To add a new element:
#   1. Append its symbol here
#   2. On next startup, the S3 bootstrap script auto-fetches properties
#      from mendeleev/pymatgen and writes them into the full registry dict
#   3. The element moves from PENDING to SUPPORTED automatically
PENDING_ELEMENTS: list[str] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_element_supported(symbol: str) -> bool:
    """Check if an element symbol is in the supported registry."""
    return symbol in SUPPORTED_ELEMENTS


def get_unsupported_elements(elements: set[str]) -> set[str]:
    """Return elements from the input set that are NOT supported."""
    return elements - SUPPORTED_ELEMENTS


def get_supported_elements_list() -> list[str]:
    """Return sorted list of supported element symbols."""
    return sorted(SUPPORTED_ELEMENTS)


def get_element_count() -> int:
    """Return total count of supported elements."""
    return len(SUPPORTED_ELEMENTS)
