"""
Piezo.AI — Central Element Registry Package
=============================================
SINGLE SOURCE OF TRUTH for supported elements.

S2: Symbol list only (for formula validation at upload).
S3: Full registry with 27+ physics properties per element,
    auto-bootstrap from mendeleev, PENDING_ELEMENTS mechanism.
"""

from piezo_ml.registry.element_registry import (
    SUPPORTED_ELEMENTS,
    PENDING_ELEMENTS,
    is_element_supported,
    get_unsupported_elements,
    get_supported_elements_list,
)

__all__ = [
    "SUPPORTED_ELEMENTS",
    "PENDING_ELEMENTS",
    "is_element_supported",
    "get_unsupported_elements",
    "get_supported_elements_list",
]
