from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _is_bad_numeric(value: Any) -> bool:
    if value is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(v) or math.isinf(v)


def validate_post_parse_frame(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if frame.empty:
        issues.append("Feature frame is empty after parsing")
        return issues

    for col in frame.columns:
        if col in {"uid", "formula"}:
            continue
        bad_count = int(sum(_is_bad_numeric(v) for v in frame[col].tolist()))
        if bad_count:
            issues.append(f"{col}: {bad_count} invalid numeric value(s)")

    frac_cols = [col for col in frame.columns if col.startswith("frac_")]
    for col in frac_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        out_of_range = int(((series < 0) | (series > 1)).sum())
        if out_of_range:
            issues.append(f"{col}: {out_of_range} value(s) out of [0, 1] range")

    for col in ("tolerance_factor", "octahedral_factor"):
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            out_of_range = int(((series < 0) | (series > 2)).sum())
            if out_of_range:
                issues.append(f"{col}: {out_of_range} value(s) out of [0, 2] range")

    var_cols = [col for col in frame.columns if col.endswith("_weighted_var")]
    for col in var_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        neg_count = int((series < 0).sum())
        if neg_count:
            issues.append(f"{col}: {neg_count} negative variance value(s)")
    return issues
