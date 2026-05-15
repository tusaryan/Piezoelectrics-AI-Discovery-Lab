"""
Piezo.AI — Central Field Options Registry
==========================================
Backward-compatible shim that delegates to field_schema_manager.

All field definitions, categories, and validation now live in
field_schema_manager.py. This module re-exports the same API so
existing imports continue to work unchanged.

Usage:
    from piezo_ml.registry.field_options_registry import (
        CATEGORICAL_OPTIONS,
        get_field_options,
        get_all_fields,
    )
"""

from __future__ import annotations

from typing import TypedDict

from piezo_ml.registry.field_schema_manager import (
    FIELD_SCHEMA,
    get_target_fields,
    get_numeric_fields as _get_numeric_fields,
    get_categorical_fields as _get_cat_fields,
    get_all_material_fields,
    get_category_values,
    is_valid_value,
)


class FieldOptionDefinition(TypedDict):
    """Definition for a categorical field's allowed values."""
    values: list[str]
    description: str
    examples: list[str]


# ---------------------------------------------------------------------------
# Build CATEGORICAL_OPTIONS dynamically from field_schema_manager
# ---------------------------------------------------------------------------

def _build_categorical_options() -> dict[str, FieldOptionDefinition]:
    """Build the CATEGORICAL_OPTIONS dict from the central field schema."""
    opts: dict[str, FieldOptionDefinition] = {}
    for name, fd in FIELD_SCHEMA.items():
        if fd.data_type == "category":
            opts[name] = {
                "values": list(fd.category_values),
                "description": fd.description,
                "examples": fd.category_values[:3] if fd.category_values else [],
            }
    return opts


CATEGORICAL_OPTIONS: dict[str, FieldOptionDefinition] = _build_categorical_options()


# ---------------------------------------------------------------------------
# Derived tuples — delegated to field_schema_manager
# ---------------------------------------------------------------------------

NUMERIC_FIELDS: tuple[str, ...] = (
    *get_target_fields(),
    *_get_numeric_fields(),
)

TARGET_FIELDS: tuple[str, ...] = get_target_fields()

ALL_MATERIAL_FIELDS: tuple[str, ...] = get_all_material_fields()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_field_options(field_name: str) -> list[str]:
    """Get allowed values for a categorical field."""
    return get_category_values(field_name)


def get_all_fields() -> list[str]:
    """Get list of all backend field names."""
    return list(ALL_MATERIAL_FIELDS)


def is_valid_categorical_value(field_name: str, value: str) -> bool:
    """Check if a value is valid for a given categorical field."""
    return is_valid_value(field_name, value)


def get_categorical_fields() -> list[str]:
    """Get list of all categorical field names."""
    return list(_get_cat_fields())