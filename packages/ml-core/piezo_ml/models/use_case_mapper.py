"""
Use-Case Mapper — maps predicted piezoelectric properties to industrial use-cases.

Rules-based classification of predicted d33, tc, and hardness values
into application categories (wearables, actuators, transducers, sensors).

Scoring uses **smooth sigmoid-like normalization** within each property's
meaningful range.  When a property is ``None`` (not predicted / user
skipped that target), its axis is excluded — the remaining axes still
produce a meaningful blend.  Confidence naturally decreases when fewer
properties are available (single-property predictions = lower certainty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class UseCaseResult:
    """Result of use-case mapping."""
    name: str
    category: str
    confidence: float  # 0.0–1.0
    description: str
    icon: str
    color: str


# Use-case thresholds and definitions
# 'ideal' = the "best-fit" centre of the ideal range for scoring
USE_CASE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "Flexible Wearable Sensors",
        "category": "wearables",
        "icon": "⌚",
        "color": "#3B82F6",
        "description": "High piezoelectric response for energy harvesting and pressure sensing in wearable devices.",
        "rules": {
            "d33": {"ideal": 300, "sigma": 200, "weight": 0.5},
            "tc": {"ideal": 150, "sigma": 150, "weight": 0.2},
            "hardness": {"ideal": 250, "sigma": 250, "weight": 0.3},
        },
    },
    {
        "name": "Industrial Actuators & Motors",
        "category": "actuators",
        "icon": "⚡",
        "color": "#F59E0B",
        "description": "High mechanical quality and thermal stability for precision actuators, ultrasonic motors, and industrial automation.",
        "rules": {
            "d33": {"ideal": 200, "sigma": 150, "weight": 0.3},
            "tc": {"ideal": 500, "sigma": 200, "weight": 0.4},
            "hardness": {"ideal": 600, "sigma": 200, "weight": 0.3},
        },
    },
    {
        "name": "Ultrasonic Transducers",
        "category": "transducers",
        "icon": "🔊",
        "color": "#EF4444",
        "description": "High mechanical hardness and thermal stability for medical ultrasound and NDT applications.",
        "rules": {
            "d33": {"ideal": 250, "sigma": 200, "weight": 0.3},
            "tc": {"ideal": 450, "sigma": 150, "weight": 0.3},
            "hardness": {"ideal": 700, "sigma": 200, "weight": 0.4},
        },
    },
    {
        "name": "Energy Harvesting Devices",
        "category": "energy",
        "icon": "🔋",
        "color": "#10B981",
        "description": "Maximum piezoelectric coefficient for converting mechanical energy to electrical energy.",
        "rules": {
            "d33": {"ideal": 500, "sigma": 200, "weight": 0.6},
            "tc": {"ideal": 300, "sigma": 200, "weight": 0.2},
            "hardness": {"ideal": 350, "sigma": 300, "weight": 0.2},
        },
    },
    {
        "name": "High-Temperature Sensors",
        "category": "ht_sensors",
        "icon": "🌡️",
        "color": "#8B5CF6",
        "description": "Excellent thermal stability (high Tc) for sensors operating in extreme temperature environments.",
        "rules": {
            "d33": {"ideal": 150, "sigma": 200, "weight": 0.2},
            "tc": {"ideal": 800, "sigma": 300, "weight": 0.6},
            "hardness": {"ideal": 500, "sigma": 300, "weight": 0.2},
        },
    },
    {
        "name": "General-Purpose Piezoelectric",
        "category": "general",
        "icon": "🎯",
        "color": "#6B7280",
        "description": "Balanced properties suitable for a wide range of piezoelectric applications.",
        "rules": {
            "d33": {"ideal": 200, "sigma": 300, "weight": 0.34},
            "tc": {"ideal": 300, "sigma": 300, "weight": 0.33},
            "hardness": {"ideal": 400, "sigma": 300, "weight": 0.33},
        },
    },
]


def _sigmoid_score(value: float, ideal: float, sigma: float) -> float:
    """Smooth Gaussian-like scoring: 1.0 at ideal, decays with distance.

    Uses  exp( -0.5 * ((value - ideal) / sigma)^2 )
    which produces a smooth bell-curve centred on *ideal* with width *sigma*.
    """
    z = (value - ideal) / max(sigma, 1.0)
    return math.exp(-0.5 * z * z)


def map_use_case(
    d33: float | None = None,
    tc: float | None = None,
    hardness: float | None = None,
) -> UseCaseResult:
    """Map predicted properties to the best-fit industrial use-case.

    Only non-None properties participate in scoring.  Confidence is
    naturally reduced when fewer properties are available.
    """
    best_score = -1.0
    best_case = USE_CASE_DEFINITIONS[-1]  # fallback: general

    for use_case in USE_CASE_DEFINITIONS[:-1]:  # skip "general" in loop
        score = _score_use_case(use_case["rules"], d33, tc, hardness)
        if score > best_score:
            best_score = score
            best_case = use_case

    # If best score is too low, fall back to general
    if best_score < 0.25:
        best_case = USE_CASE_DEFINITIONS[-1]
        best_score = _score_use_case(best_case["rules"], d33, tc, hardness)

    # Count how many properties were provided
    n_props = sum(1 for v in (d33, tc, hardness) if v is not None)

    # Apply a coverage penalty: fewer properties → lower confidence ceiling
    # 3 props: up to 1.0,  2 props: up to 0.85,  1 prop: up to 0.70
    coverage_factor = {0: 0.5, 1: 0.70, 2: 0.85, 3: 1.0}.get(n_props, 0.5)
    final_confidence = min(best_score * coverage_factor, 1.0)

    return UseCaseResult(
        name=best_case["name"],
        category=best_case["category"],
        confidence=round(final_confidence, 2),
        description=best_case["description"],
        icon=best_case["icon"],
        color=best_case["color"],
    )


def _score_use_case(
    rules: dict[str, dict[str, float]],
    d33: float | None,
    tc: float | None,
    hardness: float | None,
) -> float:
    """Score how well properties match a use-case definition.

    Uses smooth Gaussian scoring per property, weighted by property
    importance.  Properties that are None are excluded from scoring
    (their weight is redistributed to active properties).
    """
    total_weight = 0.0
    weighted_score = 0.0

    property_map = {"d33": d33, "tc": tc, "hardness": hardness}

    for prop_name, rule in rules.items():
        value = property_map.get(prop_name)
        weight = rule.get("weight", 0.33)
        if value is None:
            continue

        total_weight += weight
        ideal = rule.get("ideal", 200)
        sigma = rule.get("sigma", 300)

        score = _sigmoid_score(value, ideal, sigma)
        weighted_score += weight * score

    if total_weight <= 0:
        return 0.5

    return weighted_score / total_weight


def get_use_case_definitions() -> list[dict[str, Any]]:
    """Return all use-case definitions for frontend display."""
    return [
        {
            "name": uc["name"],
            "category": uc["category"],
            "icon": uc["icon"],
            "color": uc["color"],
            "description": uc["description"],
        }
        for uc in USE_CASE_DEFINITIONS
    ]
