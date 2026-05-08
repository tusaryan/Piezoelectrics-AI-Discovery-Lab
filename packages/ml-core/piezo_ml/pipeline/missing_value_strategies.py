from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

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

        if strategy == "drop":
            before = len(working)
            working = working[working[column].notna()].reset_index(drop=True)
            dropped_rows += before - len(working)
            continue

        if strategy == "knn":
            series = pd.to_numeric(working[column], errors="coerce")
            imputer = KNNImputer(n_neighbors=3)
            working[column] = imputer.fit_transform(series.to_frame())
            continue

        if strategy in {"mean", "median", "mode"}:
            rule = "most_frequent" if strategy == "mode" else strategy
            imputer = SimpleImputer(strategy=rule)
            working[column] = imputer.fit_transform(working[[column]]).ravel()

    return working, MissingValueReport(dropped_rows=dropped_rows, applied_strategies=strategies)
