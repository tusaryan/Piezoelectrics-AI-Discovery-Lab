from __future__ import annotations

import re
from fractions import Fraction

UNICODE_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
GARBLED_DASHES = re.compile(r"â€\"|â€\"|–|—|−|―|âˆ'")
FULLWIDTH_BRACKETS: dict[str, str] = {"（": "(", "）": ")", "（": "(", "）": ")"}
DESCRIPTIVE_TEXT = re.compile(
    r"\([^()]*(?:doped|modified|substituted|added|co-doped)[^()]*\)",
    re.IGNORECASE,
)
# Matches element symbols followed by fractional coefficients: ElementNumber/Number
# Examples: Mn1/3, Sb2/3, Sc1/2, Nb1/2, Zn1/3, Nb2/3, Mg1/3
# Pattern breakdown: (Element)(Numerator)/(Denominator)
# This handles adjacent element-fraction pairs like "Mn1/3Sb2/3" -> "Mn0.333...Sb0.666..."
FRACTION_PATTERN = re.compile(r"([A-Z][a-z]?)(\d+)/(\d+)")


def _convert_fraction_to_decimal(match: re.Match) -> str:
    """Convert fractional coefficient to decimal representation, preserving element symbol.

    Examples:
        Mn1/3 -> Mn0.3333333333333333
        Sb2/3 -> Sb0.6666666666666666
        Sc1/2 -> Sc0.5
        1/4 -> 0.25 (no element prefix)

    Args:
        match: regex match with groups: (1)=element_symbol, (2)=numerator, (3)=denominator

    Returns:
        Element symbol followed by decimal value, e.g., "Mn0.333..."
    """
    element = match.group(1)
    numerator = int(match.group(2))
    denominator = int(match.group(3))
    if denominator == 0:
        return match.group(0)  # Return original if division by zero
    fraction = Fraction(numerator, denominator)
    decimal_value = str(float(fraction))
    # Return element symbol + decimal value
    return f"{element}{decimal_value}"


def _normalize_fractions(formula: str) -> tuple[str, list[str]]:
    """Convert fractional coefficients to decimal equivalents.

    This handles formulas like:
        Mn1/3Sb2/3 -> Mn0.333...Sb0.666...
        Sc1/2Nb1/2 -> Sc0.5Nb0.5
        (Mg1/3Nb2/3)O3 -> (Mg0.333...Nb0.666...)O3
    """
    warnings: list[str] = []
    normalized = FRACTION_PATTERN.sub(_convert_fraction_to_decimal, formula)

    # Check if any fractions were converted
    if FRACTION_PATTERN.search(formula):
        warnings.append("Fractional coefficients converted to decimal")

    return normalized, warnings


def normalize_formula(formula: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    normalized = formula.strip()

    # Step 1: Convert fractional coefficients to decimal BEFORE other normalization
    # This must happen first because chemparse can't handle fractions like "1/3"
    frac_normalized, frac_warnings = _normalize_fractions(normalized)
    normalized = frac_normalized
    warnings.extend(frac_warnings)

    if any(c in normalized for c in "₀₁₂₃₄₅₆₇₈₉"):
        normalized = normalized.translate(UNICODE_SUBSCRIPTS)
        warnings.append("Unicode subscripts converted to ASCII")

    if GARBLED_DASHES.search(normalized):
        normalized = GARBLED_DASHES.sub("-", normalized)
        warnings.append("Garbled dash characters normalized")

    for fw, std in FULLWIDTH_BRACKETS.items():
        if fw in normalized:
            normalized = normalized.replace(fw, std)
    if any(x in formula for x in FULLWIDTH_BRACKETS):
        warnings.append("Full-width brackets normalized")

    match = DESCRIPTIVE_TEXT.search(normalized)
    if match:
        normalized = DESCRIPTIVE_TEXT.sub("", normalized).strip()
        warnings.append(f"Descriptive text removed: {match.group()}")

    normalized = normalized.strip("-").strip().replace(" ", "")
    return normalized, warnings