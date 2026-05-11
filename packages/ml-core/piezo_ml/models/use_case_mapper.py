"""
Use-Case Mapper — maps predicted piezoelectric properties to industrial use-cases.

Rules-based classification of predicted d33, tc, and hardness values
into application categories (wearables, actuators, transducers, sensors).
"""

from __future__ import annotations

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
USE_CASE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "Flexible Wearable Sensors",
        "category": "wearables",
        "icon": "⌚",
        "color": "#3B82F6",
        "description": "High piezoelectric response for energy harvesting and pressure sensing in wearable devices.",
        "rules": {
            "d33": {"min": 100, "max": 9999, "weight": 0.5},
            "tc": {"min": 0, "max": 300, "weight": 0.2},
            "hardness": {"min": 0, "max": 500, "weight": 0.3},
        },
    },
    {
        "name": "Industrial Actuators & Motors",
        "category": "actuators",
        "icon": "⚡",
        "color": "#F59E0B",
        "description": "High mechanical quality and thermal stability for precision actuators, ultrasonic motors, and industrial automation.",
        "rules": {
            "d33": {"min": 50, "max": 400, "weight": 0.3},
            "tc": {"min": 300, "max": 9999, "weight": 0.4},
            "hardness": {"min": 400, "max": 9999, "weight": 0.3},
        },
    },
    {
        "name": "Ultrasonic Transducers",
        "category": "transducers",
        "icon": "🔊",
        "color": "#EF4444",
        "description": "High mechanical hardness and thermal stability for medical ultrasound and NDT applications.",
        "rules": {
            "d33": {"min": 80, "max": 500, "weight": 0.3},
            "tc": {"min": 350, "max": 9999, "weight": 0.3},
            "hardness": {"min": 500, "max": 9999, "weight": 0.4},
        },
    },
    {
        "name": "Energy Harvesting Devices",
        "category": "energy",
        "icon": "🔋",
        "color": "#10B981",
        "description": "Maximum piezoelectric coefficient for converting mechanical energy to electrical energy.",
        "rules": {
            "d33": {"min": 200, "max": 9999, "weight": 0.6},
            "tc": {"min": 150, "max": 9999, "weight": 0.2},
            "hardness": {"min": 0, "max": 9999, "weight": 0.2},
        },
    },
    {
        "name": "High-Temperature Sensors",
        "category": "ht_sensors",
        "icon": "🌡️",
        "color": "#8B5CF6",
        "description": "Excellent thermal stability (high Tc) for sensors operating in extreme temperature environments.",
        "rules": {
            "d33": {"min": 40, "max": 9999, "weight": 0.2},
            "tc": {"min": 400, "max": 9999, "weight": 0.6},
            "hardness": {"min": 0, "max": 9999, "weight": 0.2},
        },
    },
    {
        "name": "General-Purpose Piezoelectric",
        "category": "general",
        "icon": "🎯",
        "color": "#6B7280",
        "description": "Balanced properties suitable for a wide range of piezoelectric applications.",
        "rules": {
            "d33": {"min": 0, "max": 9999, "weight": 0.34},
            "tc": {"min": 0, "max": 9999, "weight": 0.33},
            "hardness": {"min": 0, "max": 9999, "weight": 0.33},
        },
    },
]


def map_use_case(
    d33: float | None = None,
    tc: float | None = None,
    hardness: float | None = None,
) -> UseCaseResult:
    """Map predicted properties to the best-fit industrial use-case."""
    best_score = -1.0
    best_case = USE_CASE_DEFINITIONS[-1]  # fallback: general

    for use_case in USE_CASE_DEFINITIONS[:-1]:  # skip "general" in loop
        score = _score_use_case(use_case["rules"], d33, tc, hardness)
        if score > best_score:
            best_score = score
            best_case = use_case

    # If best score is too low, fall back to general
    if best_score < 0.3:
        best_case = USE_CASE_DEFINITIONS[-1]
        best_score = 0.5

    return UseCaseResult(
        name=best_case["name"],
        category=best_case["category"],
        confidence=round(min(best_score, 1.0), 2),
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
    """Score how well properties match a use-case definition."""
    total_weight = 0.0
    weighted_score = 0.0

    property_map = {"d33": d33, "tc": tc, "hardness": hardness}

    for prop_name, rule in rules.items():
        value = property_map.get(prop_name)
        weight = rule.get("weight", 0.33)
        if value is None:
            continue

        total_weight += weight
        min_val = rule.get("min", 0)
        max_val = rule.get("max", 9999)

        if min_val <= value <= max_val:
            # Value is in range — score based on position
            range_size = max_val - min_val
            if range_size > 0 and max_val < 9999:
                normalized = (value - min_val) / range_size
                weighted_score += weight * (0.5 + 0.5 * normalized)
            else:
                weighted_score += weight * 0.8
        elif value < min_val:
            # Below minimum — penalty proportional to distance
            penalty = max(0.0, 1.0 - abs(value - min_val) / max(min_val, 1))
            weighted_score += weight * penalty * 0.3
        else:
            # Above max — generally ok for most cases
            weighted_score += weight * 0.6

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
