"""
Sentinel Handler — detects S2 column-clear values and missing data.

During S2, column clear writes:
  - Numeric nullable → NULL (Python None)
  - Composite sentinel categoricals → "none" (valid for bulk ceramics)
  - filler_wt_pct → 0 (bulk fallback)

This module runs pre-training validation to surface issues so the user
can select per-field imputation strategies before training starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import pandas as pd

from piezo_ml.features.field_registry import CATEGORICAL_FIELDS, NUMERIC_FIELDS, TARGET_FIELDS

# Composite sentinel fields where "none" is a valid bulk-ceramic value
_COMPOSITE_SENTINEL_FIELDS = {"matrix_type", "particle_morphology", "surface_treatment"}

# Strategies that make sense for each field type
NUMERIC_STRATEGIES = ("knn", "mean", "median", "drop")
CATEGORICAL_STRATEGIES = ("mode", "drop")
TARGET_STRATEGIES = ("drop",)  # Target fields: only drop (can't impute a label)


@dataclass
class FieldIssue:
    """Describes one field's data quality issue for pre-training review."""
    field: str
    issue_type: str  # missing_numeric | missing_categorical | inconsistent_composite
    count: int
    total: int
    message: str
    suggestion: str
    default_strategy: str  # Recommended strategy
    allowed_strategies: list[str] = dc_field(default_factory=list)


def _get_allowed_strategies(field: str) -> list[str]:
    """Return the list of strategies that make sense for this field type."""
    if field in TARGET_FIELDS:
        return list(TARGET_STRATEGIES)
    if field in CATEGORICAL_FIELDS:
        return list(CATEGORICAL_STRATEGIES)
    if field in NUMERIC_FIELDS:
        return list(NUMERIC_STRATEGIES)
    # Fallback: all strategies
    return ["knn", "mean", "median", "mode", "drop"]


def detect_sentinel_issues(
    df: pd.DataFrame,
    selected_fields: list[str],
) -> list[FieldIssue]:
    """Scan selected fields for missing/sentinel values that need handling.

    Distinguishes between:
    - Legitimate sentinel values (e.g., matrix_type="none" for bulk ceramics)
    - S2-cleared/actually-missing values that need imputation or dropping
    """
    issues: list[FieldIssue] = []
    total = len(df)
    if total == 0:
        return issues

    for field in selected_fields:
        if field in ("uid", "formula"):
            continue
        if field not in df.columns:
            continue

        col = df[field]
        allowed = _get_allowed_strategies(field)

        # --- Numeric fields: check for NaN/None ---
        if field in NUMERIC_FIELDS or field in TARGET_FIELDS:
            numeric = pd.to_numeric(col, errors="coerce")
            missing = int(numeric.isna().sum())
            if missing > 0:
                is_target = field in TARGET_FIELDS
                issues.append(FieldIssue(
                    field=field,
                    issue_type="missing_numeric",
                    count=missing,
                    total=total,
                    message=(
                        f"{field}: {missing}/{total} values are missing (NULL/cleared). "
                        + ("This is a target field — rows with missing targets will be dropped."
                           if is_target else
                           "Choose an imputation strategy or drop affected rows.")
                    ),
                    suggestion="drop" if is_target else "knn",
                    default_strategy="drop" if is_target else "knn",
                    allowed_strategies=allowed,
                ))

        # --- Categorical fields ---
        elif field in CATEGORICAL_FIELDS:
            if field in _COMPOSITE_SENTINEL_FIELDS:
                # "none" is valid for bulk rows — only flag if non-bulk row has "none"
                if "filler_wt_pct" in df.columns:
                    filler_col = pd.to_numeric(df["filler_wt_pct"], errors="coerce")
                    is_bulk = (filler_col == 0) | filler_col.isna()
                    sentinel_mask = ((col == "none") | col.isna()) & ~is_bulk
                    sentinel_in_non_bulk = int(sentinel_mask.sum())
                else:
                    sentinel_in_non_bulk = 0

                if sentinel_in_non_bulk > 0:
                    issues.append(FieldIssue(
                        field=field,
                        issue_type="inconsistent_composite",
                        count=sentinel_in_non_bulk,
                        total=total,
                        message=(
                            f"{field}: {sentinel_in_non_bulk} composite rows "
                            f"have 'none'/missing value. This is inconsistent — "
                            f"composite rows need actual values."
                        ),
                        suggestion="mode",
                        default_strategy="mode",
                        allowed_strategies=allowed,
                    ))
            else:
                # Non-composite categorical — check for None/NaN/empty
                missing = int(col.isna().sum() + (col == "").sum())
                if missing > 0:
                    issues.append(FieldIssue(
                        field=field,
                        issue_type="missing_categorical",
                        count=missing,
                        total=total,
                        message=f"{field}: {missing}/{total} missing categorical values.",
                        suggestion="mode",
                        default_strategy="mode",
                        allowed_strategies=allowed,
                    ))

    return issues


def get_default_strategies(issues: list[FieldIssue]) -> dict[str, str]:
    """Return recommended strategies for each field with issues."""
    return {issue.field: issue.default_strategy for issue in issues}
