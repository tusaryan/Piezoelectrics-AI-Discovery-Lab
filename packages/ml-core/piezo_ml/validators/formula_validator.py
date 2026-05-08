"""
Piezo.AI — Formula Validator (S2 Lightweight Version)
======================================================
Validates chemical formulas against the Central Element Registry.
Used during dataset upload Review Issues step to detect:
  - Unsupported elements
  - Unicode encoding issues (subscript digits, garbled dashes)
  - Syntax problems (unmatched parentheses)
  - Empty or non-formula strings

S3 enhances this with: chemparse integration, multi-phase splitting,
stoichiometric calculation, full FormulaParser class.

See 01-architecture-and-sections.md §5.3 for formula parsing strategy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from piezo_ml.registry.element_registry import (
    SUPPORTED_ELEMENTS,
    get_unsupported_elements,
)


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

# Unicode subscript → ASCII mapping
UNICODE_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Garbled dash patterns (from XLSX → CSV conversion artifacts)
GARBLED_DASHES = re.compile(
    r"â€\"|â€\"|\u2013|\u2014|\u2212|\u2015|âˆ'"
)

# Full-width bracket replacements
FULLWIDTH_BRACKETS: dict[str, str] = {
    "（": "(",
    "）": ")",
    "\uff08": "(",
    "\uff09": ")",
}

# Element symbol pattern: uppercase letter + optional lowercase
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")

# Descriptive text in parentheses to strip
# e.g., "(Zn-Sn doped)", "(modified)", "(co-doped)"
DESCRIPTIVE_TEXT = re.compile(
    r"\([^()]*(?:doped|modified|substituted|added|co-doped)[^()]*\)",
    re.IGNORECASE,
)

# Polymer / matrix names that are NOT chemical formulas
SKIP_VALUES: frozenset[str] = frozenset({
    "pvdf", "p_vdf_trfe", "pvdf_hfp", "pvdf_hfp_ctrfe",
    "none", "na", "n/a", "",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FormulaValidationResult:
    """Structured result from formula validation."""

    formula: str                                    # Original formula string
    normalized_formula: str                         # After encoding fixes
    elements_found: set[str] = field(default_factory=set)
    unsupported_elements: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True
    error: str | None = None

    @property
    def parse_status(self) -> str:
        """Map to DB parse_status enum: pending|success|error|unsupported_elements."""
        if self.error:
            return "error"
        if self.unsupported_elements:
            return "unsupported_elements"
        return "success"

    @property
    def parse_warnings_str(self) -> str | None:
        """Semicolon-joined warnings string for DB storage."""
        if not self.warnings:
            return None
        return "; ".join(self.warnings)

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "formula": self.formula,
            "normalized_formula": self.normalized_formula,
            "elements_found": sorted(self.elements_found),
            "unsupported_elements": sorted(self.unsupported_elements),
            "warnings": self.warnings,
            "is_valid": self.is_valid,
            "error": self.error,
            "parse_status": self.parse_status,
        }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_formula(formula: str) -> tuple[str, list[str]]:
    """
    Normalize encoding issues in a formula string.

    Handles:
    - Unicode subscript digits (₀₁₂₃₄₅₆₇₈₉ → 0-9)
    - Garbled dashes from XLSX→CSV conversion
    - Full-width brackets
    - Descriptive text like "(Zn-Sn doped)"
    - Leading/trailing whitespace and dashes

    Returns:
        (normalized_formula, list_of_warnings)
    """
    warnings: list[str] = []

    # Strip whitespace
    formula = formula.strip()

    # Unicode subscripts → ASCII digits
    if any(c in formula for c in "₀₁₂₃₄₅₆₇₈₉"):
        formula = formula.translate(UNICODE_SUBSCRIPTS)
        warnings.append("Unicode subscripts converted to ASCII")

    # Garbled dashes → standard hyphen
    if GARBLED_DASHES.search(formula):
        formula = GARBLED_DASHES.sub("-", formula)
        warnings.append("Garbled dash characters normalized")

    # Full-width brackets → standard
    for fw, std in FULLWIDTH_BRACKETS.items():
        if fw in formula:
            formula = formula.replace(fw, std)
            if "Full-width brackets" not in " ".join(warnings):
                warnings.append("Full-width brackets normalized")

    # Strip descriptive text like "(Zn-Sn doped)"
    match = DESCRIPTIVE_TEXT.search(formula)
    if match:
        formula = DESCRIPTIVE_TEXT.sub("", formula).strip()
        warnings.append(f"Descriptive text removed: {match.group()}")

    # Strip trailing/leading dashes
    formula = formula.strip("-").strip()

    return formula, warnings


# ---------------------------------------------------------------------------
# Element extraction
# ---------------------------------------------------------------------------

def extract_elements(formula: str) -> set[str]:
    """
    Extract element symbols from a formula string using regex.

    This is a simple extraction — NOT stoichiometric parsing.
    It finds all patterns matching [A-Z][a-z]? which covers
    standard chemical element symbols (H, He, Li, Na, K, etc.).

    Handles formulas like:
    - KNbO3
    - (K0.5Na0.5)NbO3
    - 0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5Na0.5ZrO3

    Does NOT distinguish between elements and potential false
    positives in descriptive text — call normalize_formula first.
    """
    return {match.group() for match in ELEMENT_PATTERN.finditer(formula)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_formula(formula: str) -> FormulaValidationResult:
    """
    Validate a chemical formula against the Central Element Registry.

    Steps:
    1. Check for empty / polymer name → skip
    2. Normalize encoding (unicode, garbled dashes, brackets)
    3. Check balanced parentheses
    4. Extract element symbols
    5. Check against SUPPORTED_ELEMENTS
    6. Return structured result

    Args:
        formula: Raw formula string from CSV

    Returns:
        FormulaValidationResult with is_valid, elements, warnings, etc.
    """
    result = FormulaValidationResult(
        formula=formula,
        normalized_formula=formula,
    )

    # Empty check
    if not formula or not formula.strip():
        result.is_valid = False
        result.error = "Formula is empty"
        return result

    # Skip polymer/matrix names (not chemical formulas)
    lower = formula.strip().lower()
    if lower in SKIP_VALUES:
        result.is_valid = True
        result.warnings.append(
            "Skipped: polymer/matrix name, not a chemical formula"
        )
        return result

    # Normalize encoding
    normalized, norm_warnings = normalize_formula(formula)
    result.normalized_formula = normalized
    result.warnings.extend(norm_warnings)

    # Check balanced parentheses
    paren_depth = 0
    for char in normalized:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        if paren_depth < 0:
            result.is_valid = False
            result.error = "Unmatched closing parenthesis"
            return result
    if paren_depth != 0:
        result.is_valid = False
        result.error = "Unmatched opening parenthesis"
        return result

    # Extract elements
    elements = extract_elements(normalized)
    result.elements_found = elements

    if not elements:
        result.is_valid = False
        result.error = "No chemical elements found in formula"
        return result

    # Check against registry
    unsupported = get_unsupported_elements(elements)
    result.unsupported_elements = unsupported

    if unsupported:
        result.is_valid = False
        result.warnings.append(
            f"Unsupported elements: {', '.join(sorted(unsupported))}. "
            f"Piezo.AI currently supports {len(SUPPORTED_ELEMENTS)} elements "
            f"commonly found in perovskite piezoelectrics."
        )

    return result


def validate_formulas_batch(
    formulas: list[str],
) -> list[FormulaValidationResult]:
    """
    Validate a batch of formulas.

    Args:
        formulas: List of formula strings

    Returns:
        List of FormulaValidationResult in same order as input.
    """
    return [validate_formula(f) for f in formulas]
