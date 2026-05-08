"""Formula validation built on the S3 parser stack."""

from __future__ import annotations

from dataclasses import dataclass, field

from piezo_ml.parsers import FormulaParser, normalize_formula


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


def extract_elements(formula: str) -> set[str]:
    parser = FormulaParser()
    parsed = parser.parse(formula)
    return set(parsed.elements.keys())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_formula(formula: str) -> FormulaValidationResult:
    parser = FormulaParser()
    parsed = parser.parse(formula)
    return FormulaValidationResult(
        formula=parsed.formula,
        normalized_formula=parsed.normalized_formula,
        elements_found=set(parsed.elements.keys()),
        unsupported_elements=set(parsed.unsupported),
        warnings=parsed.warnings,
        is_valid=parsed.is_valid,
        error=parsed.error,
    )


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
