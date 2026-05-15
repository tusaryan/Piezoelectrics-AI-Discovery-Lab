"""
Piezo.AI — Central Registry Package
=====================================
SINGLE SOURCE OF TRUTH for supported elements AND field schema.

Elements: symbol list, full registry with 27+ physics properties,
          auto-bootstrap from mendeleev, PENDING_ELEMENTS mechanism.
Fields:   Central field/type/category definitions with user customizations.
"""

from piezo_ml.registry.element_registry import (
    ELEMENT_REGISTRY,
    PROPERTY_KEYS,
    SUPPORTED_ELEMENTS,
    PENDING_ELEMENTS,
    bootstrap_pending_elements,
    get_element_count,
    get_element_properties,
    is_element_supported,
    get_unsupported_elements,
    get_supported_elements_list,
)

from piezo_ml.registry.field_schema_manager import (
    FIELD_SCHEMA,
    get_field_schema,
    get_field_schema_dicts,
    get_target_fields,
    get_input_fields,
    get_categorical_fields,
    get_numeric_fields,
    get_category_values,
    get_all_material_fields,
    resolve_alias,
    is_valid_value,
    refresh_schema,
)

__all__ = [
    # Element registry
    "ELEMENT_REGISTRY",
    "PROPERTY_KEYS",
    "SUPPORTED_ELEMENTS",
    "PENDING_ELEMENTS",
    "bootstrap_pending_elements",
    "get_element_count",
    "get_element_properties",
    "is_element_supported",
    "get_unsupported_elements",
    "get_supported_elements_list",
    # Field schema
    "FIELD_SCHEMA",
    "get_field_schema",
    "get_field_schema_dicts",
    "get_target_fields",
    "get_input_fields",
    "get_categorical_fields",
    "get_numeric_fields",
    "get_category_values",
    "get_all_material_fields",
    "resolve_alias",
    "is_valid_value",
    "refresh_schema",
]

