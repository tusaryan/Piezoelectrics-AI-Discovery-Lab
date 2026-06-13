"""
Piezo.AI — Feature Field Registry
===================================
Backward-compatible shim — delegates to field_schema_manager.

All field definitions now live in piezo_ml.registry.field_schema_manager.
This module re-exports the same tuple names and helper functions so
existing imports across the ML pipeline continue to work unchanged.
"""

from __future__ import annotations

from piezo_ml.registry.field_schema_manager import (
    get_target_fields,
    get_base_input_fields,
    get_categorical_fields as _get_cat_fields,
    get_numeric_fields as _get_num_fields,
)

TARGET_FIELDS: tuple[str, ...] = get_target_fields()

BASE_INPUT_FIELDS: tuple[str, ...] = (
    "formula",
    *get_base_input_fields(),
)

CATEGORICAL_FIELDS: tuple[str, ...] = _get_cat_fields()

NUMERIC_FIELDS: tuple[str, ...] = _get_num_fields()


def get_trainable_targets(columns: list[str]) -> list[str]:
    return [target for target in TARGET_FIELDS if target in columns]


def get_feature_candidates(columns: list[str]) -> list[str]:
    return [field for field in BASE_INPUT_FIELDS if field in columns]
