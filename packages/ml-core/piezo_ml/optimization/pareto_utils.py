"""
Pareto Utilities — solution ranking, use-case tagging, and analysis helpers.

Provides use-case classification for Pareto-optimal compositions
and utilities for Pareto front analysis.
"""

from __future__ import annotations

from typing import Any


# Use-case classification rules based on predicted properties
USE_CASE_RULES: list[dict[str, Any]] = [
    {
        "tag": "Flexible Wearable",
        "color": "#3B82F6",  # blue
        "rules": {
            "d33": {"min": 200, "ideal": 400},
            "vickers_hardness": {"max": 350},
        },
    },
    {
        "tag": "Energy Harvesting",
        "color": "#10B981",  # emerald
        "rules": {
            "d33": {"min": 250, "ideal": 500},
            "tc": {"min": 200},
        },
    },
    {
        "tag": "Industrial Actuator",
        "color": "#F59E0B",  # amber
        "rules": {
            "tc": {"min": 300, "ideal": 420},
            "vickers_hardness": {"min": 300, "ideal": 600},
        },
    },
    {
        "tag": "Ultrasonic Transducer",
        "color": "#EF4444",  # red
        "rules": {
            "d33": {"min": 80},
            "tc": {"min": 350, "ideal": 450},
            "vickers_hardness": {"min": 500, "ideal": 800},
        },
    },
    {
        "tag": "High-Temp Sensor",
        "color": "#8B5CF6",  # violet
        "rules": {
            "tc": {"min": 380, "ideal": 458},
        },
    },
    {
        "tag": "Sonar/Underwater",
        "color": "#06B6D4",  # cyan
        "rules": {
            "d33": {"min": 150, "ideal": 350},
            "tc": {"min": 250},
            "vickers_hardness": {"min": 200},
        },
    },
    {
        "tag": "General Purpose",
        "color": "#6B7280",  # gray
        "rules": {},
    },
]


def tag_use_case(predicted: dict[str, float]) -> tuple[str, str]:
    """Classify a solution into the best-fit use case.

    Returns:
        (tag_label, color_hex)
    """
    best_score = -1
    best_tag = "General Purpose"
    best_color = "#6B7280"

    for rule_set in USE_CASE_RULES:
        rules = rule_set["rules"]
        if not rules:
            continue

        score = 0.0
        n_rules = 0
        all_met = True

        for prop, constraints in rules.items():
            val = predicted.get(prop)
            if val is None:
                continue
            n_rules += 1

            min_val = constraints.get("min", 0)
            max_val = constraints.get("max", float("inf"))
            ideal = constraints.get("ideal")

            if val < min_val or val > max_val:
                all_met = False
                break

            # Score based on proximity to ideal
            if ideal is not None:
                diff = abs(val - ideal) / max(ideal, 1)
                score += max(0, 1.0 - diff)
            else:
                score += 0.7  # meets constraint but no ideal defined

        if all_met and n_rules > 0 and score > best_score:
            best_score = score
            best_tag = rule_set["tag"]
            best_color = rule_set["color"]

    return best_tag, best_color


def compute_hypervolume_indicator(
    solutions: list[dict[str, float]],
    targets: list[str],
    reference_point: dict[str, float] | None = None,
) -> float:
    """Compute approximate hypervolume indicator for convergence tracking.

    Uses a simple 2D/3D bounding-box approximation.
    """
    if not solutions or not targets:
        return 0.0

    # Default reference point (worst case)
    ref = reference_point or {t: 0.0 for t in targets}

    # Compute dominated hypervolume (simplified)
    total = 0.0
    for sol in solutions:
        vol = 1.0
        for t in targets:
            val = sol.get(t, 0.0)
            r = ref.get(t, 0.0)
            vol *= max(0, val - r)
        total += vol

    return round(total, 4)


def filter_dominated(
    solutions: list[dict[str, Any]],
    targets: list[str],
    directions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Remove dominated solutions to get the true Pareto front.

    Args:
        solutions: List of solution dicts with 'predicted' key
        targets: Target property names
        directions: target -> 'maximize' or 'minimize'
    """
    if not solutions:
        return []

    dirs = directions or {t: "maximize" for t in targets}
    non_dominated = []

    for i, sol_i in enumerate(solutions):
        dominated = False
        pred_i = sol_i.get("predicted", {})

        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
            pred_j = sol_j.get("predicted", {})

            # Check if j dominates i
            all_better_or_equal = True
            at_least_one_better = False

            for t in targets:
                vi = pred_i.get(t, 0)
                vj = pred_j.get(t, 0)

                if dirs.get(t) == "maximize":
                    if vj < vi:
                        all_better_or_equal = False
                        break
                    if vj > vi:
                        at_least_one_better = True
                else:
                    if vj > vi:
                        all_better_or_equal = False
                        break
                    if vj < vi:
                        at_least_one_better = True

            if all_better_or_equal and at_least_one_better:
                dominated = True
                break

        if not dominated:
            non_dominated.append(sol_i)

    return non_dominated


def rank_solutions(
    solutions: list[dict[str, Any]],
    targets: list[str],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank solutions by weighted sum of normalized objectives.

    Args:
        solutions: List of solution dicts with 'predicted' key
        targets: Target property names
        weights: target -> weight (default 1.0 for all)
    """
    if not solutions:
        return []

    w = weights or {t: 1.0 for t in targets}

    # Normalize each target to [0, 1]
    mins = {t: float("inf") for t in targets}
    maxs = {t: float("-inf") for t in targets}

    for sol in solutions:
        pred = sol.get("predicted", {})
        for t in targets:
            val = pred.get(t, 0)
            mins[t] = min(mins[t], val)
            maxs[t] = max(maxs[t], val)

    # Compute weighted score
    for sol in solutions:
        pred = sol.get("predicted", {})
        score = 0.0
        for t in targets:
            val = pred.get(t, 0)
            rng = maxs[t] - mins[t]
            norm = (val - mins[t]) / rng if rng > 0 else 0.5
            score += norm * w.get(t, 1.0)
        sol["weighted_score"] = round(score, 4)

    return sorted(solutions, key=lambda s: s.get("weighted_score", 0), reverse=True)
