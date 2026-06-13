from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from piezo_ml.features.field_registry import CATEGORICAL_FIELDS, NUMERIC_FIELDS

MISSING_MARKERS = {"null", "none", "na", "n/a", "—", ""}


@dataclass
class CleaningReport:
    initial_rows: int
    final_rows: int
    dropped_duplicates: int


def _normalize_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in MISSING_MARKERS:
            return None
        return stripped
    return value


def clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    cleaned = frame.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].map(_normalize_cell)

    for col in NUMERIC_FIELDS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    for col in CATEGORICAL_FIELDS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string")

    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    final_rows = len(cleaned)
    return cleaned, CleaningReport(initial_rows, final_rows, initial_rows - final_rows)
