from __future__ import annotations

import re
from dataclasses import dataclass, field

import chemparse

from piezo_ml.parsers.formula_normalizer import normalize_formula
from piezo_ml.registry import get_unsupported_elements
from piezo_ml.validators.formula_strict import validate_formula_strict

SKIP_VALUES: frozenset[str] = frozenset(
    {"pvdf", "p_vdf_trfe", "pvdf_hfp", "pvdf_hfp_ctrfe", "none", "na", "n/a", ""}
)

LEADING_MULTIPLIER_RE = re.compile(r"^\s*(\d*\.?\d+)(.+)$")
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


@dataclass
class FormulaParseResult:
    formula: str
    normalized_formula: str
    elements: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    is_valid: bool = True
    error: str | None = None


def _split_top_level_phases(formula: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(formula):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unmatched closing parenthesis")
        elif char == "-" and depth == 0:
            parts.append(formula[start:idx].strip())
            start = idx + 1
    if depth != 0:
        raise ValueError("Unmatched opening parenthesis")
    parts.append(formula[start:].strip())
    return [p for p in parts if p]


def _extract_multiplier(phase: str) -> tuple[float, str]:
    match = LEADING_MULTIPLIER_RE.match(phase)
    if not match:
        return 1.0, phase
    value = float(match.group(1))
    body = match.group(2).strip()
    if not body:
        raise ValueError("Invalid phase multiplier syntax")
    return value, body


def _parse_single_phase(phase_formula: str) -> dict[str, float]:
    parsed = chemparse.parse_formula(phase_formula)
    if not parsed:
        raise ValueError("No elements found in phase")
    return {symbol: float(amount) for symbol, amount in parsed.items()}


class FormulaParser:
    def __init__(self, strict_mode: bool = False) -> None:
        self.strict_mode = strict_mode

    def parse(self, formula: str, strict_mode: bool | None = None) -> FormulaParseResult:
        """Parse a chemical formula.

        Args:
            formula: The chemical formula string.
            strict_mode: Override instance-level strict mode for this call.
                         If None, uses self.strict_mode.
        """
        use_strict = strict_mode if strict_mode is not None else self.strict_mode

        result = FormulaParseResult(formula=formula, normalized_formula=formula)
        if not formula or not formula.strip():
            result.is_valid = False
            result.error = "Formula is empty"
            return result

        lower = formula.strip().lower()
        if lower in SKIP_VALUES:
            result.warnings.append("Skipped: polymer/matrix name, not a chemical formula")
            return result

        # ── Strict pre-validation ──
        if use_strict:
            strict_result = validate_formula_strict(formula)
            if not strict_result.is_valid:
                result.is_valid = False
                result.error = "; ".join(strict_result.errors)
                result.warnings.extend(strict_result.warnings)
                return result
            result.warnings.extend(strict_result.warnings)

        normalized, warnings = normalize_formula(formula)
        result.normalized_formula = normalized
        result.warnings.extend(warnings)

        try:
            phases = _split_top_level_phases(normalized)
            totals: dict[str, float] = {}
            for phase in phases:
                multiplier, phase_formula = _extract_multiplier(phase)
                parsed = _parse_single_phase(phase_formula)
                for symbol, amount in parsed.items():
                    totals[symbol] = totals.get(symbol, 0.0) + multiplier * amount
            result.elements = totals
        except Exception as exc:
            result.is_valid = False
            result.error = str(exc)
            return result

        if not result.elements:
            result.is_valid = False
            result.error = "No chemical elements found in formula"
            return result

        symbols = set(ELEMENT_PATTERN.findall(normalized))
        unsupported = sorted(get_unsupported_elements(symbols))
        result.unsupported = unsupported
        if unsupported:
            result.is_valid = False
            result.warnings.append(f"Unsupported elements: {', '.join(unsupported)}")
        return result
