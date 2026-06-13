"""
Physics Validator — check SHAP associations against piezoelectric physics.

Validates whether the ML model's learned feature importances align
with known solid-state physics expectations for perovskite piezoelectrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysicsCheck:
    """A single physics validation check."""
    feature: str
    expected_effect: str       # "positive" | "negative" | "high_importance"
    physics_reason: str
    actual_effect: str         # "positive" | "negative" | "negligible"
    aligned: bool
    shap_magnitude: float      # mean |SHAP| for this feature
    shap_rank: int             # rank by importance (1 = most important)


@dataclass
class PhysicsValidationResult:
    """Complete physics validation result."""
    alignment_score: float      # 0-100 percentage
    total_checks: int
    confirmed: int
    violations: list[PhysicsCheck]
    confirmed_checks: list[PhysicsCheck]
    skipped: list[str]          # features not found in model


# Expected physics associations for d33 (piezoelectric coefficient)
_D33_EXPECTATIONS: list[dict[str, str]] = [
    {
        "feature": "tolerance_factor",
        "effect": "high_importance",
        "reason": "Tolerance factor near 1.0 determines perovskite phase stability — "
                  "critical for ferroelectric response. Top-ranked by physics theory.",
    },
    {
        "feature": "en_pauling_weighted_mean",
        "effect": "positive",
        "reason": "Higher average electronegativity increases bond ionicity, "
                  "enhancing spontaneous polarization and d33.",
    },
    {
        "feature": "ionic_radius_pm_weighted_var",
        "effect": "positive",
        "reason": "High ionic radius variance indicates morphotropic phase boundary (MPB) "
                  "proximity — MPB compositions show maximum d33.",
    },
    {
        "feature": "atomic_mass_weighted_mean",
        "effect": "positive",
        "reason": "Heavier atoms create slower lattice dynamics, favoring "
                  "larger piezoelectric distortions at low frequencies.",
    },
    {
        "feature": "valence_electrons_weighted_mean",
        "effect": "positive",
        "reason": "More valence electrons enable stronger covalent-ionic bonding "
                  "character, contributing to lattice polarizability.",
    },
    {
        "feature": "octahedral_factor",
        "effect": "high_importance",
        "reason": "Octahedral factor (rB/rO) determines B-site ion fit — "
                  "too small leads to tilting, reducing polarization.",
    },
]

# Expected associations for tc (Curie temperature)
_TC_EXPECTATIONS: list[dict[str, str]] = [
    {
        "feature": "tolerance_factor",
        "effect": "negative",
        "reason": "Higher tolerance factor (closer to cubic) generally reduces Tc — "
                  "tetragonal distortion correlates with higher Tc.",
    },
    {
        "feature": "melting_point_k_weighted_mean",
        "effect": "positive",
        "reason": "Higher melting point indicates stronger bonding, "
                  "typically associated with higher Curie temperatures.",
    },
    {
        "feature": "en_pauling_weighted_mean",
        "effect": "positive",
        "reason": "Stronger ionic character (higher EN) produces deeper "
                  "ferroelectric energy wells, raising Tc.",
    },
    {
        "feature": "ionic_radius_pm_weighted_var",
        "effect": "negative",
        "reason": "MPB compositions (high radius variance) tend to have "
                  "lower Tc due to phase instability.",
    },
]

# Expected associations for vickers_hardness
_HARDNESS_EXPECTATIONS: list[dict[str, str]] = [
    {
        "feature": "bulk_modulus_gpa_weighted_mean",
        "effect": "positive",
        "reason": "Higher bulk modulus directly correlates with higher "
                  "resistance to permanent deformation (hardness).",
    },
    {
        "feature": "shear_modulus_gpa_weighted_mean",
        "effect": "positive",
        "reason": "Shear modulus relates to Pugh's ratio (G/B), "
                  "a reliable predictor of brittle (hard) behavior.",
    },
    {
        "feature": "en_pauling_weighted_mean",
        "effect": "positive",
        "reason": "Higher electronegativity → more covalent character → "
                  "harder materials (Gilman-Chin correlation).",
    },
]

_TARGET_EXPECTATIONS = {
    "d33": _D33_EXPECTATIONS,
    "tc": _TC_EXPECTATIONS,
    "vickers_hardness": _HARDNESS_EXPECTATIONS,
}


class PhysicsValidator:
    """Validate SHAP importances against known piezoelectric physics."""

    def validate(
        self,
        feature_names: list[str],
        mean_abs_shap: list[float],
        shap_values_matrix: list[list[float]] | np.ndarray | None,
        target: str = "d33",
    ) -> PhysicsValidationResult:
        """Run physics validation checks."""
        expectations = _TARGET_EXPECTATIONS.get(target, _D33_EXPECTATIONS)

        # Build importance ranking
        importance = dict(zip(feature_names, mean_abs_shap))
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        rank_map = {feat: i + 1 for i, (feat, _) in enumerate(sorted_feats)}

        # Compute directional effects if we have the full SHAP matrix
        direction_map: dict[str, str] = {}
        if shap_values_matrix is not None:
            sv = np.array(shap_values_matrix) if not isinstance(shap_values_matrix, np.ndarray) else shap_values_matrix
            for i, feat in enumerate(feature_names):
                if i < sv.shape[1]:
                    mean_shap = float(np.mean(sv[:, i]))
                    if abs(mean_shap) < 1e-6:
                        direction_map[feat] = "negligible"
                    elif mean_shap > 0:
                        direction_map[feat] = "positive"
                    else:
                        direction_map[feat] = "negative"

        confirmed_list: list[PhysicsCheck] = []
        violation_list: list[PhysicsCheck] = []
        skipped: list[str] = []

        for exp in expectations:
            feat = exp["feature"]
            if feat not in importance:
                skipped.append(feat)
                continue

            mag = importance[feat]
            rank = rank_map[feat]
            actual_dir = direction_map.get(feat, "unknown")
            expected = exp["effect"]

            if expected == "high_importance":
                # Check if feature is in top 40% by importance
                threshold = max(1, int(len(feature_names) * 0.4))
                aligned = rank <= threshold
            elif expected in ("positive", "negative"):
                aligned = actual_dir == expected or actual_dir == "negligible"
            else:
                aligned = True

            check = PhysicsCheck(
                feature=feat,
                expected_effect=expected,
                physics_reason=exp["reason"],
                actual_effect=actual_dir,
                aligned=aligned,
                shap_magnitude=round(mag, 6),
                shap_rank=rank,
            )
            if aligned:
                confirmed_list.append(check)
            else:
                violation_list.append(check)

        total = len(confirmed_list) + len(violation_list)
        score = (len(confirmed_list) / total * 100) if total > 0 else 0.0

        return PhysicsValidationResult(
            alignment_score=round(score, 1),
            total_checks=total,
            confirmed=len(confirmed_list),
            violations=violation_list,
            confirmed_checks=confirmed_list,
            skipped=skipped,
        )
