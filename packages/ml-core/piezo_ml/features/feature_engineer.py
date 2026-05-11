from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from piezo_ml.parsers import FormulaParser
from piezo_ml.registry import ELEMENT_REGISTRY, PROPERTY_KEYS

WEIGHTED_PROPERTIES: tuple[str, ...] = tuple(
    key
    for key in PROPERTY_KEYS
    if key not in {"symbol", "oxidation_states", "block", "perovskite_site", "is_rare_earth"}
)


@dataclass
class EngineeredRow:
    uid: int
    formula: str
    normalized_formula: str
    element_amounts: dict[str, float]
    element_fractions: dict[str, float]
    weighted_features: dict[str, float]
    warnings: list[str]


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def _weighted_stats(values: list[tuple[float, float]]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    total_w = sum(weight for _, weight in values)
    if total_w <= 0:
        return 0.0, 0.0
    mean = sum(v * w for v, w in values) / total_w
    variance = sum(w * (v - mean) ** 2 for v, w in values) / total_w
    return mean, variance


class FeatureEngineer:
    def __init__(self) -> None:
        self.parser = FormulaParser()

    def engineer_row(self, uid: int, formula: str) -> EngineeredRow:
        parsed = self.parser.parse(formula)
        if not parsed.is_valid:
            raise ValueError(parsed.error or "; ".join(parsed.warnings) or "Invalid formula")

        element_amounts = dict(parsed.elements)
        non_oxygen_total = sum(v for k, v in element_amounts.items() if k != "O")
        denominator = non_oxygen_total if non_oxygen_total > 0 else sum(element_amounts.values())
        if denominator <= 0:
            raise ValueError("Parsed formula has zero stoichiometric amount")

        fractions = {
            symbol: amount / denominator
            for symbol, amount in element_amounts.items()
            if symbol != "O"
        }
        if not fractions:
            fractions = {symbol: amount / denominator for symbol, amount in element_amounts.items()}

        weighted_features: dict[str, float] = {}
        for prop in WEIGHTED_PROPERTIES:
            values: list[tuple[float, float]] = []
            for symbol, fraction in fractions.items():
                raw = ELEMENT_REGISTRY[symbol].get(prop)
                num = _numeric_or_none(raw)
                if num is not None:
                    values.append((num, fraction))
            mean, variance = _weighted_stats(values)
            weighted_features[f"{prop}_weighted_mean"] = mean
            weighted_features[f"{prop}_weighted_var"] = variance

        weighted_features.update(self._compute_structural_factors(element_amounts))

        return EngineeredRow(
            uid=uid,
            formula=formula,
            normalized_formula=parsed.normalized_formula,
            element_amounts=element_amounts,
            element_fractions=fractions,
            weighted_features=weighted_features,
            warnings=parsed.warnings,
        )

    def _compute_structural_factors(self, element_amounts: dict[str, float]) -> dict[str, float]:
        oxygen_radius = _numeric_or_none(ELEMENT_REGISTRY.get("O", {}).get("ionic_radius_pm"))
        a_values: list[tuple[float, float]] = []
        b_values: list[tuple[float, float]] = []

        for symbol, amount in element_amounts.items():
            if symbol == "O":
                continue
            props = ELEMENT_REGISTRY[symbol]
            radius = _numeric_or_none(props.get("ionic_radius_pm"))
            if radius is None:
                continue
            site = props.get("perovskite_site")
            if site == "A":
                a_values.append((radius, amount))
            elif site == "B":
                b_values.append((radius, amount))

        r_a, _ = _weighted_stats(a_values)
        r_b, _ = _weighted_stats(b_values)
        if oxygen_radius is None or oxygen_radius <= 0 or r_a <= 0 or r_b <= 0:
            return {"tolerance_factor": 0.0, "octahedral_factor": 0.0}

        tolerance = (r_a + oxygen_radius) / (math.sqrt(2.0) * (r_b + oxygen_radius))
        octahedral = r_b / oxygen_radius
        return {"tolerance_factor": tolerance, "octahedral_factor": octahedral}

    def engineer_dataframe(
        self,
        frame: pd.DataFrame,
        formula_col: str = "formula",
        uid_col: str = "uid",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        compositions: list[dict[str, Any]] = []
        skipped_uids: list[tuple[int, str]] = []  # (uid, reason)

        # Columns to carry over from source for the unified parsed dataset
        _CARRY_OVER_COLS = [
            "d33", "tc", "vickers_hardness", "qm", "kp",
            "relative_density_pct", "sintering_temp_c",
            "sintering_method", "ceramic_type", "fabrication_method",
            "matrix_type", "filler_wt_pct", "particle_morphology",
            "particle_size_nm", "surface_treatment",
        ]

        for _, row in frame.iterrows():
            uid = int(row[uid_col])
            formula_raw = str(row[formula_col])

            # Skip empty/null formulas
            if not formula_raw or formula_raw.lower() in ("nan", "none", ""):
                skipped_uids.append((uid, "empty formula"))
                continue

            try:
                engineered = self.engineer_row(uid=uid, formula=formula_raw)
            except ValueError as e:
                skipped_uids.append((uid, str(e)))
                continue

            base = {"uid": engineered.uid, "formula": engineered.normalized_formula}
            base.update({f"frac_{k}": v for k, v in engineered.element_fractions.items()})
            base.update(engineered.weighted_features)
            rows.append(base)

            comp = {
                "uid": engineered.uid,
                "formula": engineered.normalized_formula,
                "parse_status": "success",
                "parse_warnings": "; ".join(engineered.warnings) if engineered.warnings else "",
            }
            # Add element amounts
            comp.update(engineered.element_amounts)
            # Carry over original material properties for unified artifact
            for col in _CARRY_OVER_COLS:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        comp[col] = val
            compositions.append(comp)

        # Store skipped info for callers to access
        self._last_skipped_uids = skipped_uids

        vectors_df = pd.DataFrame(rows).fillna(0.0)
        parsed_df = pd.DataFrame(compositions).fillna(0.0)
        return vectors_df, parsed_df

