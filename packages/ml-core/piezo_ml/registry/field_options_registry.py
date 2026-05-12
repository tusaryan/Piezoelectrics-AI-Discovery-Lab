"""
Piezo.AI — Central Field Options Registry
==========================================
SINGLE SOURCE OF TRUTH for categorical field allowed values.

This module centralizes ALL allowed values for categorical fields
used across the entire platform (API, ML-Core, Frontend).

Usage:
    from piezo_ml.registry.field_options_registry import (
        CATEGORICAL_OPTIONS,
        get_field_options,
        get_all_fields,
    )
"""

from __future__ import annotations

from typing import TypedDict


class FieldOptionDefinition(TypedDict):
    """Definition for a categorical field's allowed values."""
    values: list[str]
    description: str
    examples: list[str]


# ---------------------------------------------------------------------------
# Central categorical field options registry
# ---------------------------------------------------------------------------
# All allowed values for every categorical field in the platform.
# Used by: API schemas, ML preprocessing, formula validation, frontend dropdowns.

CATEGORICAL_OPTIONS: dict[str, FieldOptionDefinition] = {
    "sintering_method": {
        "values": [
            "conventional", "hot_press", "sps", "rtgg", "tgg",
            "two_step", "cold_sinter", "microwave", "flash",
        ],
        "description": "Sintering technique used for ceramic fabrication",
        "examples": ["conventional", "hot_press", "sps"],
    },
    "ceramic_type": {
        "values": ["soft", "hard", "composite"],
        "description": "PZT ceramic type classification based on doping",
        "examples": ["soft", "hard"],
    },
    "fabrication_method": {
        "values": [
            "conventional", "hot_press", "sps", "rtgg", "tgg", "two_step",
            "electrospinning", "solvent_cast", "cold_sinter", "hot_compression",
            "3d_print", "tape_casting", "screen_printing", "injection_molding",
        ],
        "description": "Primary ceramic or composite fabrication technique",
        "examples": ["conventional", "solvent_cast", "electrospinning"],
    },
    "matrix_type": {
        "values": [
            "none", "pvdf", "p_vdf_trfe", "pvdf_trfe", "pvdf_hfp",
            "pvdf_hfp_ctrfe", "epoxy", "silicone", "polyimide", "pdms",
        ],
        "description": "Polymer matrix for composite materials (none=bulk ceramic)",
        "examples": ["none", "pvdf", "pvdf_hfp"],
    },
    "particle_morphology": {
        "values": [
            "none", "spherical", "rod", "cube", "nanoblock",
            "fiber", "platelet", "whisker", "irregular",
        ],
        "description": "Shape/geometry of ceramic filler particles",
        "examples": ["spherical", "fiber", "platelet"],
    },
    "surface_treatment": {
        "values": [
            "none", "untreated", "silane", "plasma", "acid",
            "peg", "dopamine", "fluorinated", "kh550", "kh560",
        ],
        "description": "Surface modification applied to filler particles",
        "examples": ["untreated", "silane", "plasma"],
    },
}


# ---------------------------------------------------------------------------
# Numeric fields (for reference and validation)
# ---------------------------------------------------------------------------

NUMERIC_FIELDS: tuple[str, ...] = (
    "d33", "tc", "vickers_hardness",
    "qm", "kp", "relative_density_pct", "sintering_temp_c",
    "filler_wt_pct", "particle_size_nm",
)

# Target fields (can be used as ML training targets)
TARGET_FIELDS: tuple[str, ...] = ("d33", "tc", "vickers_hardness")

# All material fields (for CRUD and mapping)
ALL_MATERIAL_FIELDS: tuple[str, ...] = (
    "formula",
    "d33", "tc", "vickers_hardness",
    "qm", "kp", "relative_density_pct", "sintering_temp_c",
    "sintering_method", "ceramic_type", "fabrication_method",
    "matrix_type", "filler_wt_pct", "particle_morphology",
    "particle_size_nm", "surface_treatment",
    "source_doi", "source_notes",
)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_field_options(field_name: str) -> list[str]:
    """Get allowed values for a categorical field."""
    return CATEGORICAL_OPTIONS.get(field_name, {}).get("values", [])


def get_all_fields() -> list[str]:
    """Get list of all backend field names."""
    return list(ALL_MATERIAL_FIELDS)


def is_valid_categorical_value(field_name: str, value: str) -> bool:
    """Check if a value is valid for a given categorical field."""
    return value in CATEGORICAL_OPTIONS.get(field_name, {}).get("values", [])


def get_categorical_fields() -> list[str]:
    """Get list of all categorical field names."""
    return list(CATEGORICAL_OPTIONS.keys())