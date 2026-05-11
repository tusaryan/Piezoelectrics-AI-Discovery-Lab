"""
Piezo.AI — Validators Package
===============================
Formula validation and data quality checks.

S2: Basic formula validator (element extraction + registry check).
S3: Full FormulaParser with chemparse, multi-phase splitting,
    stoichiometric calculation, and post-parse validation.
S5: Strict formula validator (bracket rules, charset, element patterns).
"""

# NOTE: formula_validator.py imports from piezo_ml.parsers, which imports
# from piezo_ml.validators.formula_strict. To break the circular import,
# we import formula_strict first (it has no circular deps), then defer
# formula_validator imports.

from piezo_ml.validators.formula_strict import (
    validate_formula_strict,
    StrictValidationResult,
)


def __getattr__(name):
    """Lazy import to break circular dependency with parsers module."""
    if name in (
        "validate_formula", "validate_formulas_batch",
        "normalize_formula", "extract_elements",
        "FormulaValidationResult",
    ):
        from piezo_ml.validators.formula_validator import (
            validate_formula,
            validate_formulas_batch,
            normalize_formula,
            extract_elements,
            FormulaValidationResult,
        )
        _lazy = {
            "validate_formula": validate_formula,
            "validate_formulas_batch": validate_formulas_batch,
            "normalize_formula": normalize_formula,
            "extract_elements": extract_elements,
            "FormulaValidationResult": FormulaValidationResult,
        }
        return _lazy[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "validate_formula",
    "validate_formulas_batch",
    "normalize_formula",
    "extract_elements",
    "FormulaValidationResult",
    "validate_formula_strict",
    "StrictValidationResult",
]
