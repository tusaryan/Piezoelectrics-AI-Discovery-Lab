"""
Usage Prediction Engine — rule-based scoring for piezoelectric use cases.

Takes any subset of {d33, Tc, Hardness} and recommends suitable real-world
applications. Grounded in peer-reviewed literature, USPTO patents, and
industry sources (see Project/usage-engineering-logic.md).

Implements:
- 11 use-case categories with property-driven scoring (0–100)
- Confidence tiers: Primary (≥70), Secondary (45–69), Tertiary (30–44)
- Composite vs bulk ceramic modifiers
- Partial-property handling with scaling and caution notes
- Scientific caution note generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Icons for use cases ──
USE_CASE_ICONS: dict[str, str] = {
    "Medical Ultrasound Imaging": "🏥",
    "Implantable Biomedical Devices": "🫀",
    "NDT / Industrial Sensors": "🏭",
    "High-Power Ultrasonics": "⚡",
    "Automotive Sensors": "🚗",
    "Aerospace / High-Temp SHM": "✈️",
    "Energy Harvesting": "🔋",
    "Sonar / Underwater Acoustics": "🌊",
    "Wearable / IoT Sensors": "⌚",
    "Extreme Environment Sensing": "☢️",
    "Precision Actuators / MEMS": "🔬",
}

USE_CASE_COLORS: dict[str, str] = {
    "Medical Ultrasound Imaging": "#EC4899",
    "Implantable Biomedical Devices": "#F43F5E",
    "NDT / Industrial Sensors": "#3B82F6",
    "High-Power Ultrasonics": "#F59E0B",
    "Automotive Sensors": "#EF4444",
    "Aerospace / High-Temp SHM": "#6366F1",
    "Energy Harvesting": "#10B981",
    "Sonar / Underwater Acoustics": "#06B6D4",
    "Wearable / IoT Sensors": "#8B5CF6",
    "Extreme Environment Sensing": "#DC2626",
    "Precision Actuators / MEMS": "#14B8A6",
}


@dataclass
class UseCaseResult:
    """A single use-case recommendation."""
    name: str
    score: int                     # 0–100 raw score
    confidence: float              # 0.0–1.0 normalized
    tier: str                      # "primary" | "secondary" | "tertiary"
    tier_label: str                # "Highly Recommended" | "Good Fit" | "Possible"
    description: str               # 1-sentence rationale
    icon: str
    color: str
    driving_properties: list[str]  # which properties contributed
    category: str = ""             # backward compat


@dataclass
class UsagePredictionResult:
    """Complete usage prediction output."""
    recommendations: list[UseCaseResult]
    caution_notes: list[str]
    property_completeness: str     # "full" | "partial" | "minimal"
    properties_used: list[str]
    is_composite: bool = False


def predict_usage(
    d33: float | None = None,
    tc: float | None = None,
    hardness: float | None = None,
    is_composite: bool = False,
) -> UsagePredictionResult:
    """
    Main entry point: predict suitable applications from material properties.

    Args:
        d33: Piezoelectric coefficient (pC/N), or None if not predicted.
        tc: Curie temperature (°C), or None if not predicted.
        hardness: Vickers hardness (HV), or None if not predicted.
        is_composite: Whether material is a polymer-ceramic composite.

    Returns:
        UsagePredictionResult with ranked recommendations and caution notes.
    """
    scores = _compute_scores(d33, tc, hardness)

    # Apply composite modifiers
    if is_composite:
        _apply_composite_modifiers(scores)

    # Determine property completeness
    props_used = []
    if d33 is not None:
        props_used.append("d33")
    if tc is not None:
        props_used.append("tc")
    if hardness is not None:
        props_used.append("hardness")

    n_props = len(props_used)
    if n_props == 3:
        completeness = "full"
    elif n_props >= 1:
        completeness = "partial"
        # Scale scores up proportionally for partial data
        scale = 100.0 / _max_possible_score(n_props)
        for key in scores:
            scores[key] = min(100, int(scores[key] * scale))
    else:
        completeness = "minimal"

    # Build recommendations (filter ≥ 30)
    recommendations = []
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score < 30:
            continue

        tier, tier_label = _classify_tier(score)
        description = _get_description(name, d33, tc, hardness)
        driving = _get_driving_properties(name, d33, tc, hardness)

        recommendations.append(UseCaseResult(
            name=name,
            score=score,
            confidence=round(score / 100.0, 2),
            tier=tier,
            tier_label=tier_label,
            description=description,
            icon=USE_CASE_ICONS.get(name, "🔧"),
            color=USE_CASE_COLORS.get(name, "#6366F1"),
            driving_properties=driving,
            category=name.lower().replace(" ", "_").replace("/", "_"),
        ))

    # Limit to top 5
    recommendations = recommendations[:5]

    # Generate caution notes
    cautions = _generate_cautions(d33, tc, hardness)

    return UsagePredictionResult(
        recommendations=recommendations,
        caution_notes=cautions,
        property_completeness=completeness,
        properties_used=props_used,
        is_composite=is_composite,
    )


def _compute_scores(
    d33: float | None, tc: float | None, hv: float | None,
) -> dict[str, int]:
    """Core scoring logic from usage-engineering-logic.md Part 3.4."""
    scores: dict[str, int] = {}

    # ── MEDICAL IMAGING / BIOMEDICAL ──
    s = 0
    if d33 is not None:
        if d33 >= 500:
            s += 40
        elif d33 >= 200:
            s += 20
    if tc is not None:
        if 100 <= tc <= 350:
            s += 25
    if hv is not None:
        if hv < 600:
            s += 15
    scores["Medical Ultrasound Imaging"] = s

    # ── IMPLANTABLE ENERGY HARVESTER / BIOMEDICAL ──
    s = 0
    if d33 is not None:
        if d33 >= 1000:
            s += 45
        elif d33 >= 500:
            s += 25
    if tc is not None:
        if 100 <= tc <= 250:
            s += 25
    if hv is not None:
        if hv < 400:
            s += 20
    scores["Implantable Biomedical Devices"] = s

    # ── NDT / INDUSTRIAL SENSORS ──
    s = 0
    if d33 is not None:
        if 50 <= d33 < 500:
            s += 30
    if tc is not None:
        if 200 <= tc <= 500:
            s += 35
    if hv is not None:
        if 500 <= hv <= 1000:
            s += 25
    scores["NDT / Industrial Sensors"] = s

    # ── HIGH-POWER ULTRASONICS (welding/cleaning) ──
    s = 0
    if d33 is not None:
        if d33 >= 200:
            s += 25
    if tc is not None:
        if tc >= 300:
            s += 30
    if hv is not None:
        if hv >= 700:
            s += 35
    scores["High-Power Ultrasonics"] = s

    # ── AUTOMOTIVE SENSORS ──
    s = 0
    if d33 is not None:
        if 50 <= d33 < 500:
            s += 25
    if tc is not None:
        if tc >= 350:
            s += 40
    if hv is not None:
        if hv >= 600:
            s += 25
    scores["Automotive Sensors"] = s

    # ── AEROSPACE / HIGH-TEMP STRUCTURAL HEALTH MONITORING ──
    s = 0
    if d33 is not None:
        if d33 >= 100:
            s += 20
    if tc is not None:
        if tc >= 450:
            s += 45
    if hv is not None:
        if hv >= 800:
            s += 25
    scores["Aerospace / High-Temp SHM"] = s

    # ── ENERGY HARVESTING (ambient vibration) ──
    s = 0
    if d33 is not None:
        if d33 >= 150:
            s += 40
    if tc is not None:
        if tc >= 200:
            s += 25
    if hv is not None:
        s += 15  # hardness less critical for energy harvesting
    scores["Energy Harvesting"] = s

    # ── SONAR / UNDERWATER ACOUSTICS ──
    s = 0
    if d33 is not None:
        if 50 <= d33 <= 600:
            s += 30
    if tc is not None:
        if tc >= 250:
            s += 25
    if hv is not None:
        if hv >= 700:
            s += 35
    scores["Sonar / Underwater Acoustics"] = s

    # ── WEARABLE / IoT ──
    s = 0
    if d33 is not None:
        if d33 >= 100:
            s += 30
    if tc is not None:
        if 100 <= tc <= 250:
            s += 20
    if hv is not None:
        if hv < 400:
            s += 35
    scores["Wearable / IoT Sensors"] = s

    # ── EXTREME ENVIRONMENT (nuclear/geothermal) ──
    s = 0
    if d33 is not None:
        if 10 <= d33 < 100:
            s += 20
    if tc is not None:
        if tc >= 600:
            s += 55
    if hv is not None:
        if hv >= 900:
            s += 20
    scores["Extreme Environment Sensing"] = s

    # ── PRECISION ACTUATORS (MEMS / nanopositioning) ──
    s = 0
    if d33 is not None:
        if d33 >= 300:
            s += 40
    if tc is not None:
        if tc >= 200:
            s += 25
    if hv is not None:
        if 400 <= hv <= 900:
            s += 25
    scores["Precision Actuators / MEMS"] = s

    return scores


def _max_possible_score(n_props: int) -> float:
    """Approximate max score achievable with n properties (for scaling)."""
    # With 3 properties, max single use-case score ≈ 90
    # With 2 properties, max ≈ 65–70
    # With 1 property, max ≈ 40–55
    if n_props >= 3:
        return 90.0
    elif n_props == 2:
        return 65.0
    else:
        return 45.0


def _apply_composite_modifiers(scores: dict[str, int]) -> None:
    """Apply composite material modifiers (Part 6 of reference)."""
    # Boost wearable + implantable
    scores["Wearable / IoT Sensors"] = scores.get("Wearable / IoT Sensors", 0) + 15
    scores["Implantable Biomedical Devices"] = scores.get("Implantable Biomedical Devices", 0) + 15
    scores["Medical Ultrasound Imaging"] = scores.get("Medical Ultrasound Imaging", 0) + 10
    # Boost underwater (better acoustic impedance matching)
    scores["Sonar / Underwater Acoustics"] = scores.get("Sonar / Underwater Acoustics", 0) + 10
    # Reduce high-power ultrasonics (mechanical fragility)
    scores["High-Power Ultrasonics"] = max(0, scores.get("High-Power Ultrasonics", 0) - 20)
    # Cap at 100
    for k in scores:
        scores[k] = min(100, scores[k])


def _classify_tier(score: int) -> tuple[str, str]:
    """Map score to confidence tier."""
    if score >= 70:
        return "primary", "Highly Recommended"
    elif score >= 45:
        return "secondary", "Good Fit"
    elif score >= 30:
        return "tertiary", "Possible Application"
    return "excluded", ""


# ── Description templates ──

_DESCRIPTIONS: dict[str, str] = {
    "Medical Ultrasound Imaging":
        "High piezoelectric sensitivity enables precise acoustic signal generation for diagnostic imaging.",
    "Implantable Biomedical Devices":
        "Exceptional piezo response at body-safe temperatures suits deep brain stimulation and implant energy harvesting.",
    "NDT / Industrial Sensors":
        "Balanced properties suit accelerometers, flow meters, and structural health monitoring in factories.",
    "High-Power Ultrasonics":
        "Hard ceramic with thermal stability endures continuous high-power duty cycles for welding and cleaning.",
    "Automotive Sensors":
        "High Curie temperature withstands engine bay heat; suitable for knock, pressure, and fuel injection sensors.",
    "Aerospace / High-Temp SHM":
        "Extreme thermal stability enables structural monitoring near jet engines and in aerospace applications.",
    "Energy Harvesting":
        "Strong charge generation from ambient vibrations converts mechanical energy to electrical power.",
    "Sonar / Underwater Acoustics":
        "Durable ceramic with moderate sensitivity handles shock-intensive underwater acoustic environments.",
    "Wearable / IoT Sensors":
        "Low hardness and good sensitivity enable flexible, skin-conformable sensors for health monitoring.",
    "Extreme Environment Sensing":
        "Ultra-high Curie temperature allows sensing in nuclear, geothermal, and turbine-adjacent environments.",
    "Precision Actuators / MEMS":
        "High charge coefficient drives nanometer-precision positioning for microscopy and semiconductor tools.",
}


def _get_description(name: str, d33, tc, hv) -> str:
    return _DESCRIPTIONS.get(name, "")


def _get_driving_properties(name: str, d33, tc, hv) -> list[str]:
    """Determine which properties drove this recommendation."""
    drivers = []
    if d33 is not None:
        drivers.append(f"d₃₃={d33:.0f} pC/N")
    if tc is not None:
        drivers.append(f"Tc={tc:.0f}°C")
    if hv is not None:
        drivers.append(f"HV={hv:.0f}")
    return drivers


def _generate_cautions(d33, tc, hv) -> list[str]:
    """Generate scientific caution notes based on property values."""
    cautions = []

    if tc is not None and tc < 200:
        cautions.append(
            "Low Curie temperature limits deployment to controlled "
            f"environments (max operating temp ≈{tc/2:.0f}°C)."
        )
    if d33 is not None and d33 < 50:
        cautions.append(
            "Low piezoelectric coefficient restricts use to sensing; "
            "not suitable for high-displacement actuation."
        )
    if hv is not None and hv < 300:
        cautions.append(
            "Low hardness indicates flexible/polymer-class material; "
            "unsuitable for rigid high-load mechanical environments."
        )
    if d33 is not None and d33 > 1000 and tc is not None and tc < 150:
        cautions.append(
            "High d₃₃ + low Tc (PMN-PT class) offers exceptional sensitivity "
            "but requires thermal management for field deployment."
        )

    return cautions


# ── Backward-compatible wrapper ──

def map_use_case(
    d33: float | None = None,
    tc: float | None = None,
    hardness: float | None = None,
    is_composite: bool = False,
) -> UseCaseResult:
    """
    Backward-compatible wrapper: returns the top use-case recommendation.

    Used by prediction service for the 'Suggested Use Case' card.
    """
    result = predict_usage(d33=d33, tc=tc, hardness=hardness, is_composite=is_composite)

    if result.recommendations:
        return result.recommendations[0]

    # Fallback: general purpose
    return UseCaseResult(
        name="General Purpose Piezoelectric",
        score=20,
        confidence=0.20,
        tier="tertiary",
        tier_label="General",
        description="Insufficient property data for specific application matching.",
        icon="🔧",
        color="#6B7280",
        driving_properties=[],
        category="general",
    )


# Also export for advanced UI
def get_use_case_definitions() -> dict[str, dict]:
    """Return all use-case definitions for UI display."""
    return {
        name: {"icon": icon, "color": USE_CASE_COLORS.get(name, "#6366F1")}
        for name, icon in USE_CASE_ICONS.items()
    }
