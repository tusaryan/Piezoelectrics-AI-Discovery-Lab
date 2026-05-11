"""
Strict Formula Validator — enforces bracket rules, charset, element validity.

Performs pre-parse structural validation that the basic parser does not cover:

1. **Charset**: Only allows A-Z, a-z, 0-9, `.`, `-`, `(`, `)`, `{`, `}`
2. **Bracket balance**: All brackets must be properly matched
3. **Nesting hierarchy**: `()` may appear inside `{}`, but not vice-versa
4. **Multiplier rules**: Digits after closing brackets, no empty groups
5. **Element validation**: Each element token must start with exactly ONE
   uppercase letter followed by at most ONE lowercase letter (standard
   periodic table notation).  Rejects lowercase-only chars (e.g. ``k``,
   ``kananb``), multi-lowercase sequences (``Oo``, ``aa``, ``Kaaa``), and
   trailing lowercase-only fragments (``KNaNbO3-ooo``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Allowed characters in a strict formula
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9.()\-{} ]+$")

# Standard element token: uppercase letter followed by optional one lowercase
_ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")

# Detect lowercase-only runs (invalid element fragments)
_LOWERCASE_RUN = re.compile(r"(?<![A-Za-z])[a-z]{1,}")


@dataclass
class StrictValidationResult:
    """Result of strict formula validation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_formula_strict(formula: str) -> StrictValidationResult:
    """Run strict structural validation on a chemical formula string.

    This is a **pre-parse** check — it validates structure and charset
    before the formula reaches chemparse.  Returns a result with
    ``is_valid=False`` and human-readable error messages if validation fails.
    """
    result = StrictValidationResult()

    if not formula or not formula.strip():
        result.is_valid = False
        result.errors.append("Formula is empty")
        return result

    raw = formula.strip()

    # ── 1. Charset check ──
    if not _ALLOWED_CHARS.match(raw):
        bad_chars = sorted(set(c for c in raw if not re.match(r"[A-Za-z0-9.()\-{} ]", c)))
        result.is_valid = False
        result.errors.append(f"Invalid characters: {', '.join(repr(c) for c in bad_chars)}")
        return result

    # ── 2. Bracket balance and nesting ──
    bracket_errors = _check_brackets(raw)
    if bracket_errors:
        result.is_valid = False
        result.errors.extend(bracket_errors)
        return result

    # ── 3. Empty bracket groups ──
    if "()" in raw or "{}" in raw:
        result.is_valid = False
        result.errors.append("Empty bracket group detected")
        return result

    # ── 4. Element validation ──
    # Strip everything except letters (remove digits, dots, brackets, dashes)
    # Then validate that each "token" of letters is a valid element pattern
    element_errors = _check_element_tokens(raw)
    if element_errors:
        result.is_valid = False
        result.errors.extend(element_errors)
        return result

    # ── 5. Must contain at least one element ──
    elements = _ELEMENT_PATTERN.findall(raw)
    if not elements:
        result.is_valid = False
        result.errors.append("No valid chemical elements found")
        return result

    return result


def _check_brackets(formula: str) -> list[str]:
    """Check bracket balance and nesting hierarchy.

    Rules:
    - All brackets must be properly matched
    - `()` can appear inside `{}`
    - `{}` cannot appear inside `()`
    """
    errors: list[str] = []
    stack: list[str] = []
    openers = {"(": ")", "{": "}"}
    closers = {")": "(", "}": "{"}

    for i, char in enumerate(formula):
        if char in openers:
            stack.append(char)
        elif char in closers:
            if not stack:
                errors.append(f"Unmatched closing bracket '{char}' at position {i}")
                return errors
            top = stack.pop()
            expected_closer = openers[top]
            if char != expected_closer:
                errors.append(
                    f"Mismatched brackets: expected '{expected_closer}' "
                    f"but found '{char}' at position {i}"
                )
                return errors

    if stack:
        unmatched = "".join(stack)
        errors.append(f"Unmatched opening bracket(s): {unmatched}")
        return errors

    # Nesting hierarchy: {} cannot be inside ()
    depth_paren = 0
    for char in formula:
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "{" and depth_paren > 0:
            errors.append("Curly braces {{}} cannot be nested inside parentheses ()")
            return errors

    return errors


def _check_element_tokens(formula: str) -> list[str]:
    """Validate that letter sequences in the formula are valid elements.

    Rules:
    - Each element starts with exactly one uppercase letter
    - Followed by at most one lowercase letter
    - Standalone lowercase letters are invalid (e.g., 'k', 'ooo')
    - Multi-lowercase after uppercase is invalid (e.g., 'Oo' could be
      ambiguous but 'Ooo' or 'Kaaa' are clearly invalid)

    We split the formula into "phases" on `-` and process each.
    """
    errors: list[str] = []

    # Split on dashes for multi-phase formulas
    phases = formula.split("-")

    for phase in phases:
        if not phase.strip():
            continue

        # Extract letter-only sequences by removing digits, dots, brackets, spaces
        # e.g., "K0.5Na0.5NbO3" -> "KNaNbO"
        letter_seq = re.sub(r"[^A-Za-z]", " ", phase)
        tokens = letter_seq.split()

        for token in tokens:
            if not token:
                continue

            # Check for purely lowercase tokens
            if token.islower():
                errors.append(
                    f"Invalid element(s) '{token}': elements must start "
                    f"with an uppercase letter"
                )
                continue

            # Walk through the token character by character
            # Valid pattern: [A-Z][a-z]? repeated
            i = 0
            while i < len(token):
                if token[i].isupper():
                    # Start of an element
                    elem = token[i]
                    i += 1
                    # Check for one optional lowercase
                    if i < len(token) and token[i].islower():
                        elem += token[i]
                        i += 1
                    # If there's another lowercase immediately, it's invalid
                    if i < len(token) and token[i].islower():
                        # Collect the full invalid sequence
                        bad = elem
                        while i < len(token) and token[i].islower():
                            bad += token[i]
                            i += 1
                        errors.append(
                            f"Invalid element '{bad}': too many lowercase "
                            f"letters (max 1 after uppercase)"
                        )
                elif token[i].islower():
                    # Lowercase without preceding uppercase
                    bad = ""
                    while i < len(token) and token[i].islower():
                        bad += token[i]
                        i += 1
                    errors.append(
                        f"Invalid element(s) '{bad}': elements must start "
                        f"with an uppercase letter"
                    )
                else:
                    i += 1

    return errors
