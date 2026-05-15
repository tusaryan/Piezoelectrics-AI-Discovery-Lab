"""
Piezo.AI — Central Field Schema Manager
=========================================
SINGLE SOURCE OF TRUTH for all material fields, their data types,
allowed categories, value ranges, aliases, and metadata.

This is the **field-level** equivalent of element_registry.py.
Every module that needs to know about fields imports from here.

Usage:
    from piezo_ml.registry.field_schema_manager import (
        FIELD_SCHEMA,
        get_field_schema,
        get_target_fields,
        get_input_fields,
        get_categorical_fields,
        get_numeric_fields,
        get_category_values,
        resolve_alias,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dc_field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIELD_CUSTOMIZATIONS_PATH = PROJECT_ROOT / "resources" / ".field-customizations.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FieldDefinition:
    """Definition for a single material data field."""
    name: str
    data_type: str                               # "float" | "int" | "string" | "category"
    description: str = ""
    is_target: bool = False                      # Can be ML target
    is_input: bool = True                        # Can be ML input feature
    is_required: bool = False                    # Required in dataset
    is_composite_field: bool = False             # Relevant only for composites
    category_values: list[str] = dc_field(default_factory=list)
    aliases: dict[str, str] = dc_field(default_factory=dict)
    range_min: float | None = None
    range_max: float | None = None
    default_value: str | None = None
    is_user_added: bool = False
    added_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldDefinition":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Default field schema — hardcoded app defaults
# ---------------------------------------------------------------------------

DEFAULT_FIELD_SCHEMA: dict[str, FieldDefinition] = {
    # ── Formula (special) ──
    "formula": FieldDefinition(
        name="formula",
        data_type="string",
        description="Chemical formula of the material (e.g., BaTiO3, KNaNbO3)",
        is_target=False,
        is_input=True,
        is_required=True,
    ),

    # ── Target fields ──
    "d33": FieldDefinition(
        name="d33",
        data_type="float",
        description="Piezoelectric coefficient d₃₃ (pC/N)",
        is_target=True,
        is_input=True,
        range_min=0,
        range_max=3000,
    ),
    "tc": FieldDefinition(
        name="tc",
        data_type="float",
        description="Curie temperature Tc (°C)",
        is_target=True,
        is_input=True,
        range_min=-50,
        range_max=1500,
    ),
    "vickers_hardness": FieldDefinition(
        name="vickers_hardness",
        data_type="float",
        description="Vickers hardness (HV)",
        is_target=True,
        is_input=True,
        range_min=0,
        range_max=2000,
    ),

    # ── Numeric input fields ──
    "qm": FieldDefinition(
        name="qm",
        data_type="float",
        description="Mechanical quality factor Qm",
        range_min=0,
    ),
    "kp": FieldDefinition(
        name="kp",
        data_type="float",
        description="Planar electromechanical coupling factor kp",
        range_min=0,
        range_max=1,
    ),
    "relative_density_pct": FieldDefinition(
        name="relative_density_pct",
        data_type="float",
        description="Relative density (%)",
        range_min=0,
        range_max=100,
    ),
    "sintering_temp_c": FieldDefinition(
        name="sintering_temp_c",
        data_type="float",
        description="Sintering temperature (°C)",
        range_min=0,
        range_max=2000,
    ),
    "filler_wt_pct": FieldDefinition(
        name="filler_wt_pct",
        data_type="float",
        description="Filler weight percentage for composites (%)",
        is_composite_field=True,
        range_min=0,
        range_max=100,
        default_value="0",
    ),
    "particle_size_nm": FieldDefinition(
        name="particle_size_nm",
        data_type="float",
        description="Particle size in nanometers",
        is_composite_field=True,
        range_min=0,
    ),

    # ── Categorical input fields ──
    "sintering_method": FieldDefinition(
        name="sintering_method",
        data_type="category",
        description="Sintering technique used for ceramic fabrication",
        category_values=[
            "conventional", "hot_press", "sps", "rtgg", "tgg",
            "two_step", "cold_sinter", "microwave", "flash",
            "spark_plasma", "hot_pressing", "cold_sintering",
            "liquid_phase", "none",
        ],
    ),
    "ceramic_type": FieldDefinition(
        name="ceramic_type",
        data_type="category",
        description="PZT ceramic type classification based on doping",
        category_values=["soft", "hard", "composite"],
    ),
    "fabrication_method": FieldDefinition(
        name="fabrication_method",
        data_type="category",
        description="Primary ceramic or composite fabrication technique",
        category_values=[
            "conventional", "hot_press", "sps", "rtgg", "tgg", "two_step",
            "electrospinning", "solvent_cast", "cold_sinter", "hot_compression",
            "3d_print", "tape_casting", "screen_printing", "injection_molding",
            "bridgman", "flux_growth", "hydrothermal", "sputtering",
            "czochralski", "solid_state", "sol_gel", "solution_cast",
            "spin_coating", "3d_printing",
        ],
    ),
    "matrix_type": FieldDefinition(
        name="matrix_type",
        data_type="category",
        description="Polymer matrix for composite materials (none = bulk ceramic)",
        is_composite_field=True,
        category_values=[
            "none", "pvdf", "p_vdf_trfe", "pvdf_trfe", "pvdf_hfp",
            "pvdf_hfp_ctrfe", "epoxy", "silicone", "polyimide", "pdms",
            "polyurea", "plla", "pla", "polyurethane",
        ],
        aliases={
            "pvdf-trfe": "p_vdf_trfe",
            "p(vdf-trfe)": "p_vdf_trfe",
        },
        default_value="none",
    ),
    "particle_morphology": FieldDefinition(
        name="particle_morphology",
        data_type="category",
        description="Shape/geometry of ceramic filler particles",
        is_composite_field=True,
        category_values=[
            "none", "spherical", "rod", "cube", "nanoblock",
            "fiber", "platelet", "whisker", "irregular",
            "nanowire", "nanosheet", "tube", "unknown",
        ],
        default_value="none",
    ),
    "surface_treatment": FieldDefinition(
        name="surface_treatment",
        data_type="category",
        description="Surface modification applied to filler particles",
        is_composite_field=True,
        category_values=[
            "none", "untreated", "silane", "plasma", "acid",
            "peg", "dopamine", "fluorinated", "kh550", "kh560",
            "oleic_acid", "hydrogen_peroxide",
        ],
        aliases={
            "untreated": "none",
        },
        default_value="none",
    ),

    # ── Metadata fields (not used as ML features) ──
    "source_doi": FieldDefinition(
        name="source_doi",
        data_type="string",
        description="DOI of the source publication",
        is_input=False,
    ),
    "source_notes": FieldDefinition(
        name="source_notes",
        data_type="string",
        description="Notes about the data source",
        is_input=False,
    ),
}


# ---------------------------------------------------------------------------
# Customization persistence
# ---------------------------------------------------------------------------

def _load_customizations() -> dict[str, Any]:
    """Load user-added field customizations."""
    if not FIELD_CUSTOMIZATIONS_PATH.exists():
        return {"added_fields": {}, "added_categories": {}, "added_aliases": {}}
    try:
        with open(FIELD_CUSTOMIZATIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure required keys exist
        data.setdefault("added_fields", {})
        data.setdefault("added_categories", {})
        data.setdefault("added_aliases", {})
        return data
    except Exception:
        return {"added_fields": {}, "added_categories": {}, "added_aliases": {}}


def _save_customizations(data: dict[str, Any]) -> None:
    """Save user-added field customizations."""
    FIELD_CUSTOMIZATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIELD_CUSTOMIZATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Schema assembly — merge defaults + user customizations
# ---------------------------------------------------------------------------

def _build_field_schema() -> dict[str, FieldDefinition]:
    """Build the complete field schema by merging defaults with user customizations."""
    schema = {name: FieldDefinition(**asdict(fd)) for name, fd in DEFAULT_FIELD_SCHEMA.items()}
    customs = _load_customizations()

    # Merge user-added categories into existing fields
    for field_name, extra_cats in customs.get("added_categories", {}).items():
        if field_name in schema and schema[field_name].data_type == "category":
            existing = set(schema[field_name].category_values)
            for cat in extra_cats:
                if cat not in existing:
                    schema[field_name].category_values.append(cat)

    # Merge user-added aliases into existing fields
    for field_name, extra_aliases in customs.get("added_aliases", {}).items():
        if field_name in schema:
            schema[field_name].aliases.update(extra_aliases)

    # Add user-created fields
    for field_name, field_data in customs.get("added_fields", {}).items():
        if field_name not in schema:
            fd = FieldDefinition.from_dict(field_data)
            fd.is_user_added = True
            schema[field_name] = fd

    return schema


# Module-level schema — refreshed on demand
FIELD_SCHEMA: dict[str, FieldDefinition] = _build_field_schema()


def refresh_schema() -> None:
    """Rebuild FIELD_SCHEMA from defaults + customizations."""
    global FIELD_SCHEMA
    FIELD_SCHEMA = _build_field_schema()


# ---------------------------------------------------------------------------
# Read accessors — used by all other modules
# ---------------------------------------------------------------------------

def get_field_schema() -> dict[str, FieldDefinition]:
    """Return the complete merged field schema."""
    return FIELD_SCHEMA


def get_field_schema_dicts() -> list[dict[str, Any]]:
    """Return field schema as a list of dicts (for API serialization)."""
    return [fd.to_dict() for fd in FIELD_SCHEMA.values()]


def get_target_fields() -> tuple[str, ...]:
    """Return names of fields that can be ML targets."""
    return tuple(name for name, fd in FIELD_SCHEMA.items() if fd.is_target)


def get_input_fields() -> tuple[str, ...]:
    """Return names of fields that can be ML input features (excluding formula)."""
    return tuple(
        name for name, fd in FIELD_SCHEMA.items()
        if fd.is_input and name != "formula" and not fd.is_target
    )


def get_base_input_fields() -> tuple[str, ...]:
    """Return all ML-eligible input fields including formula."""
    return tuple(name for name, fd in FIELD_SCHEMA.items() if fd.is_input and not fd.is_target)


def get_categorical_fields() -> tuple[str, ...]:
    """Return names of categorical fields."""
    return tuple(name for name, fd in FIELD_SCHEMA.items() if fd.data_type == "category")


def get_numeric_fields() -> tuple[str, ...]:
    """Return names of numeric input fields (excludes targets, formula, categories)."""
    return tuple(
        name for name, fd in FIELD_SCHEMA.items()
        if fd.data_type in ("float", "int") and fd.is_input and not fd.is_target
    )


def get_category_values(field_name: str) -> list[str]:
    """Get allowed category values for a field."""
    fd = FIELD_SCHEMA.get(field_name)
    if fd and fd.data_type == "category":
        return list(fd.category_values)
    return []


def get_all_material_fields() -> tuple[str, ...]:
    """Return all field names in order."""
    return tuple(FIELD_SCHEMA.keys())


def get_composite_fields() -> tuple[str, ...]:
    """Return names of composite-specific fields."""
    return tuple(name for name, fd in FIELD_SCHEMA.items() if fd.is_composite_field)


def resolve_alias(field_name: str, value: str) -> str:
    """Resolve an alias to its canonical value for a field."""
    fd = FIELD_SCHEMA.get(field_name)
    if fd and fd.aliases:
        return fd.aliases.get(value.lower().strip(), value)
    return value


def is_valid_value(field_name: str, value: str) -> bool:
    """Check if a value is valid for a categorical field."""
    fd = FIELD_SCHEMA.get(field_name)
    if not fd or fd.data_type != "category":
        return True  # Non-categorical fields don't have value restrictions
    resolved = resolve_alias(field_name, value)
    return resolved in fd.category_values


# ---------------------------------------------------------------------------
# Write operations — for user customizations
# ---------------------------------------------------------------------------

def add_user_field(field_data: dict[str, Any]) -> dict[str, Any]:
    """Add a new user-defined field to the schema."""
    import re
    name = field_data.get("name", "").strip()
    if not name or not re.match(r'^[a-z][a-z0-9_]*$', name):
        return {"error": "Field name must be snake_case (e.g., dielectric_constant)"}
    if name in DEFAULT_FIELD_SCHEMA:
        return {"error": f"'{name}' is a built-in field and cannot be overridden"}
    if name in FIELD_SCHEMA:
        return {"error": f"'{name}' already exists in the schema"}

    data_type = field_data.get("data_type", "float")
    if data_type not in ("float", "int", "string", "category"):
        return {"error": f"Invalid data_type '{data_type}'. Must be: float, int, string, category"}

    field_data["is_user_added"] = True
    field_data["added_at"] = datetime.now(timezone.utc).isoformat()

    customs = _load_customizations()
    customs["added_fields"][name] = field_data
    _save_customizations(customs)
    refresh_schema()

    return {"message": f"Field '{name}' added successfully", "field_name": name}


def remove_user_field(name: str) -> dict[str, Any]:
    """Remove a user-added field from the schema."""
    if name in DEFAULT_FIELD_SCHEMA:
        return {"error": f"'{name}' is a built-in field and cannot be removed"}

    customs = _load_customizations()
    if name not in customs.get("added_fields", {}):
        return {"error": f"'{name}' is not a user-added field"}

    del customs["added_fields"][name]
    # Also clean up any added categories/aliases
    customs.get("added_categories", {}).pop(name, None)
    customs.get("added_aliases", {}).pop(name, None)
    _save_customizations(customs)
    refresh_schema()

    return {"message": f"Field '{name}' removed"}


def add_category_value(field_name: str, value: str) -> dict[str, Any]:
    """Add a new category value to an existing categorical field."""
    fd = FIELD_SCHEMA.get(field_name)
    if not fd:
        return {"error": f"Field '{field_name}' not found"}
    if fd.data_type != "category":
        return {"error": f"Field '{field_name}' is not a categorical field"}

    value = value.strip().lower().replace(" ", "_")
    if not value:
        return {"error": "Category value cannot be empty"}
    if value in fd.category_values:
        return {"error": f"'{value}' already exists in '{field_name}'"}

    customs = _load_customizations()
    added_cats = customs.setdefault("added_categories", {})
    field_cats = added_cats.setdefault(field_name, [])
    field_cats.append(value)
    _save_customizations(customs)
    refresh_schema()

    return {"message": f"Category '{value}' added to '{field_name}'", "value": value}


def remove_category_value(field_name: str, value: str) -> dict[str, Any]:
    """Remove a user-added category value. Cannot remove built-in values."""
    fd = FIELD_SCHEMA.get(field_name)
    if not fd:
        return {"error": f"Field '{field_name}' not found"}

    # Check if it's a user-added category
    customs = _load_customizations()
    added_cats = customs.get("added_categories", {}).get(field_name, [])
    if value not in added_cats:
        return {"error": f"'{value}' is a built-in category and cannot be removed"}

    added_cats.remove(value)
    if not added_cats:
        customs.get("added_categories", {}).pop(field_name, None)
    else:
        customs["added_categories"][field_name] = added_cats
    _save_customizations(customs)
    refresh_schema()

    return {"message": f"Category '{value}' removed from '{field_name}'"}


def add_alias(field_name: str, alias: str, canonical: str) -> dict[str, Any]:
    """Add an alias mapping for a field."""
    fd = FIELD_SCHEMA.get(field_name)
    if not fd:
        return {"error": f"Field '{field_name}' not found"}

    alias = alias.strip().lower()
    canonical = canonical.strip().lower()
    if not alias or not canonical:
        return {"error": "Alias and canonical value cannot be empty"}
    if fd.data_type == "category" and canonical not in fd.category_values:
        return {"error": f"Canonical value '{canonical}' is not a valid category for '{field_name}'"}

    customs = _load_customizations()
    added_aliases = customs.setdefault("added_aliases", {})
    field_aliases = added_aliases.setdefault(field_name, {})
    field_aliases[alias] = canonical
    _save_customizations(customs)
    refresh_schema()

    return {"message": f"Alias '{alias}' → '{canonical}' added to '{field_name}'"}


def export_field_schema() -> dict[str, Any]:
    """Export the full field schema as a portable JSON structure."""
    return {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "fields": {name: fd.to_dict() for name, fd in FIELD_SCHEMA.items()},
        "customizations": _load_customizations(),
    }


def import_field_schema(data: dict[str, Any]) -> dict[str, Any]:
    """Import field customizations from an exported schema."""
    if "customizations" not in data:
        return {"error": "Invalid schema format: missing 'customizations' key"}

    customs = data["customizations"]
    _save_customizations(customs)
    refresh_schema()

    n_fields = len(customs.get("added_fields", {}))
    n_cats = sum(len(v) for v in customs.get("added_categories", {}).values())
    n_aliases = sum(len(v) for v in customs.get("added_aliases", {}).values())

    return {
        "message": f"Imported {n_fields} fields, {n_cats} categories, {n_aliases} aliases",
        "fields_imported": n_fields,
        "categories_imported": n_cats,
        "aliases_imported": n_aliases,
    }


def reset_field_schema() -> dict[str, Any]:
    """Remove all user customizations and reset to defaults."""
    if FIELD_CUSTOMIZATIONS_PATH.exists():
        FIELD_CUSTOMIZATIONS_PATH.unlink()
    refresh_schema()
    return {"message": "Field schema reset to defaults"}
