from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from piezo_ml.features.field_registry import CATEGORICAL_FIELDS

StrategyName = Literal["knn", "mean", "median", "mode", "drop"]


@dataclass
class MissingValueReport:
    dropped_rows: int
    applied_strategies: dict[str, StrategyName]


def apply_missing_value_strategies(
    frame: pd.DataFrame,
    strategies: dict[str, StrategyName],
) -> tuple[pd.DataFrame, MissingValueReport]:
    working = frame.copy()
    dropped_rows = 0

    for column, strategy in strategies.items():
        if column not in working.columns:
            continue

        is_categorical = column in CATEGORICAL_FIELDS

        # Normalize NA types: convert pd.NA → np.nan / None so sklearn can handle them
        if is_categorical:
            working[column] = working[column].astype(object).where(working[column].notna(), None)
        else:
            working[column] = pd.to_numeric(working[column], errors="coerce")

        if strategy == "drop":
            before = len(working)
            working = working[working[column].notna()].reset_index(drop=True)
            dropped_rows += before - len(working)
            continue

        if strategy == "mode":
            # Mode: works for both categorical and numeric
            # For categorical, fill NaN/None with the most frequent non-null value
            mode_val = working[column].dropna().mode()
            if len(mode_val) > 0:
                working[column] = working[column].fillna(mode_val.iloc[0])
            continue

        if strategy == "knn":
            if is_categorical:
                # KNN can't handle categorical — fall back to mode
                mode_val = working[column].dropna().mode()
                if len(mode_val) > 0:
                    working[column] = working[column].fillna(mode_val.iloc[0])
            else:
                # KNN for numeric columns only
                series = pd.to_numeric(working[column], errors="coerce")
                imputer = KNNImputer(n_neighbors=3)
                working[column] = imputer.fit_transform(series.to_frame())
            continue

        if strategy in {"mean", "median"}:
            if is_categorical:
                # Mean/median can't handle categorical — fall back to mode
                mode_val = working[column].dropna().mode()
                if len(mode_val) > 0:
                    working[column] = working[column].fillna(mode_val.iloc[0])
            else:
                fill_val = working[column].mean() if strategy == "mean" else working[column].median()
                if pd.notna(fill_val):
                    working[column] = working[column].fillna(fill_val)

    return working, MissingValueReport(dropped_rows=dropped_rows, applied_strategies=strategies)
