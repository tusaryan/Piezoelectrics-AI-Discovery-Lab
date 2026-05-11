"""
Piezo.AI — Validators Package
===============================
Formula validation and data quality checks.

S2: Basic formula validator (element extraction + registry check).
S3: Full FormulaParser with chemparse, multi-phase splitting,
    stoichiometric calculation, and post-parse validation.
S5: Strict formula validator (bracket rules, charset, element patterns).
"""

from piezo_ml.validators.formula_validator import (
    validate_formula,
    validate_formulas_batch,
    normalize_formula,
    extract_elements,
    FormulaValidationResult,
)

from piezo_ml.validators.formula_strict import (
    validate_formula_strict,
    StrictValidationResult,
)

__all__ = [
    "validate_formula",
    "validate_formulas_batch",
    "normalize_formula",
    "extract_elements",
    "FormulaValidationResult",
    "validate_formula_strict",
    "StrictValidationResult",
]

