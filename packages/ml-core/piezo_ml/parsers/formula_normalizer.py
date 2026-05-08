from __future__ import annotations

import re

UNICODE_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
GARBLED_DASHES = re.compile(r"â€\"|â€\"|\u2013|\u2014|\u2212|\u2015|âˆ'")
FULLWIDTH_BRACKETS: dict[str, str] = {"（": "(", "）": ")", "\uff08": "(", "\uff09": ")"}
DESCRIPTIVE_TEXT = re.compile(
    r"\([^()]*(?:doped|modified|substituted|added|co-doped)[^()]*\)",
    re.IGNORECASE,
)


def normalize_formula(formula: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    normalized = formula.strip()

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

    normalized = normalized.strip("-").strip()
    return normalized, warnings
