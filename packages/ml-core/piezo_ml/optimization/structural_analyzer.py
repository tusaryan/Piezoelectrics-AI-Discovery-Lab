"""
Structural Analyzer — Pymatgen-based crystal structure analysis.

Computes physics-based structural descriptors for perovskite materials:
- Tolerance factor (Goldschmidt)
- Octahedral factor
- Bond valence estimates
- Crystal system classification
- Structural stability indicators
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from piezo_ml.features import FeatureEngineer
from piezo_ml.parsers import FormulaParser
from piezo_ml.registry import ELEMENT_REGISTRY


@dataclass
class StructuralDescriptor:
    """Full structural analysis result for a composition."""
    formula: str
    normalized_formula: str
    is_valid: bool = True
    error: str | None = None

    # Goldschmidt criteria
    tolerance_factor: float = 0.0
    octahedral_factor: float = 0.0
    crystal_system: str = "unknown"
    stability_class: str = "unknown"

    # Bond-valence estimates
    avg_bond_valence_a: float = 0.0
    avg_bond_valence_b: float = 0.0
    bond_valence_mismatch: float = 0.0

    # Elemental composition summary
    a_site_elements: dict[str, float] = field(default_factory=dict)
    b_site_elements: dict[str, float] = field(default_factory=dict)
    dopant_elements: dict[str, float] = field(default_factory=dict)
    oxygen_content: float = 0.0
    total_elements: int = 0

    # Physics descriptors
    avg_electronegativity: float = 0.0
    electronegativity_diff: float = 0.0
    avg_atomic_mass: float = 0.0
    avg_ionic_radius_a: float = 0.0
    avg_ionic_radius_b: float = 0.0
    polarizability_index: float = 0.0

    # Size-mismatch descriptor (lattice strain)
    a_site_variance: float = 0.0
    b_site_variance: float = 0.0

    # Additional perovskite indicators
    is_perovskite_likely: bool = False
    perovskite_confidence: float = 0.0
    phase_count: int = 1
    warnings: list[str] = field(default_factory=list)


def _safe_float(value: Any) -> float | None:
    """Extract numeric value safely."""
    if value is None:
        return None
    try:
        v = float(value)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except (TypeError, ValueError):
        return None


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Compute weighted mean from (value, weight) pairs."""
    if not pairs:
        return 0.0
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_w


def _weighted_variance(pairs: list[tuple[float, float]]) -> float:
    """Compute weighted variance from (value, weight) pairs."""
    if not pairs:
        return 0.0
    mean = _weighted_mean(pairs)
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(w * (v - mean) ** 2 for v, w in pairs) / total_w


class StructuralAnalyzer:
    """Analyze crystal structure from chemical formula using physics-based descriptors."""

    def __init__(self) -> None:
        self.parser = FormulaParser()
        self.engineer = FeatureEngineer()

    def analyze(self, formula: str) -> StructuralDescriptor:
        """Run full structural analysis for a single formula."""
        parsed = self.parser.parse(formula)
        if not parsed.is_valid:
            return StructuralDescriptor(
                formula=formula,
                normalized_formula=formula,
                is_valid=False,
                error=parsed.error or "; ".join(parsed.warnings) or "Parse failed",
            )

        elements = dict(parsed.elements)
        descriptor = StructuralDescriptor(
            formula=formula,
            normalized_formula=parsed.normalized_formula,
            total_elements=len(elements),
            phase_count=max(1, len(parsed.phases) if hasattr(parsed, "phases") else 1),
        )

        # Classify elements by perovskite site
        self._classify_sites(elements, descriptor)

        # Compute structural factors
        self._compute_goldschmidt(descriptor)

        # Compute bond-valence estimates
        self._compute_bond_valence(descriptor)

        # Compute electronic/physics descriptors
        self._compute_physics_descriptors(elements, descriptor)

        # Classify crystal system from tolerance factor
        self._classify_crystal_system(descriptor)

        # Assess perovskite likelihood
        self._assess_perovskite(elements, descriptor)

        return descriptor

    def compare(self, formulas: list[str]) -> list[StructuralDescriptor]:
        """Analyze and compare multiple compositions."""
        return [self.analyze(f) for f in formulas]

    def _classify_sites(
        self, elements: dict[str, float], desc: StructuralDescriptor,
    ) -> None:
        """Classify elements into A-site, B-site, oxygen, dopant."""
        for symbol, amount in elements.items():
            if symbol == "O":
                desc.oxygen_content = amount
                continue
            props = ELEMENT_REGISTRY.get(symbol, {})
            site = props.get("perovskite_site", "dopant")
            if site == "A":
                desc.a_site_elements[symbol] = amount
            elif site == "B":
                desc.b_site_elements[symbol] = amount
            else:
                desc.dopant_elements[symbol] = amount

    def _compute_goldschmidt(self, desc: StructuralDescriptor) -> None:
        """Compute Goldschmidt tolerance and octahedral factors."""
        o_radius = _safe_float(ELEMENT_REGISTRY.get("O", {}).get("ionic_radius_pm"))

        a_radii = []
        for sym, amt in desc.a_site_elements.items():
            r = _safe_float(ELEMENT_REGISTRY.get(sym, {}).get("ionic_radius_pm"))
            if r is not None:
                a_radii.append((r, amt))

        b_radii = []
        for sym, amt in desc.b_site_elements.items():
            r = _safe_float(ELEMENT_REGISTRY.get(sym, {}).get("ionic_radius_pm"))
            if r is not None:
                b_radii.append((r, amt))

        r_a = _weighted_mean(a_radii)
        r_b = _weighted_mean(b_radii)
        desc.avg_ionic_radius_a = round(r_a, 2)
        desc.avg_ionic_radius_b = round(r_b, 2)

        # Variance for lattice-strain descriptor
        desc.a_site_variance = round(_weighted_variance(a_radii), 4)
        desc.b_site_variance = round(_weighted_variance(b_radii), 4)

        if o_radius and o_radius > 0 and r_a > 0 and r_b > 0:
            desc.tolerance_factor = round(
                (r_a + o_radius) / (math.sqrt(2.0) * (r_b + o_radius)), 4,
            )
            desc.octahedral_factor = round(r_b / o_radius, 4)
        else:
            desc.warnings.append("Missing ionic radius data for Goldschmidt calculation")

    def _compute_bond_valence(self, desc: StructuralDescriptor) -> None:
        """Estimate bond valence sums for A and B sites."""
        # Use oxidation states as proxy for expected bond valence
        a_valences = []
        for sym, amt in desc.a_site_elements.items():
            ox = ELEMENT_REGISTRY.get(sym, {}).get("oxidation_states", [])
            if ox and isinstance(ox, list):
                # Take the most common positive oxidation state
                positive = [x for x in ox if isinstance(x, (int, float)) and x > 0]
                if positive:
                    a_valences.append((float(positive[0]), amt))

        b_valences = []
        for sym, amt in desc.b_site_elements.items():
            ox = ELEMENT_REGISTRY.get(sym, {}).get("oxidation_states", [])
            if ox and isinstance(ox, list):
                positive = [x for x in ox if isinstance(x, (int, float)) and x > 0]
                if positive:
                    b_valences.append((float(max(positive)), amt))

        desc.avg_bond_valence_a = round(_weighted_mean(a_valences), 2)
        desc.avg_bond_valence_b = round(_weighted_mean(b_valences), 2)

        # Bond valence mismatch: deviation from ideal ABO3 (A=2+, B=4+ or 5+)
        ideal_a = 2.0
        ideal_b = 5.0  # For niobates
        mismatch_a = abs(desc.avg_bond_valence_a - ideal_a) if desc.avg_bond_valence_a > 0 else 0
        mismatch_b = abs(desc.avg_bond_valence_b - ideal_b) if desc.avg_bond_valence_b > 0 else 0
        desc.bond_valence_mismatch = round(mismatch_a + mismatch_b, 3)

    def _compute_physics_descriptors(
        self, elements: dict[str, float], desc: StructuralDescriptor,
    ) -> None:
        """Compute electronic and mass-based descriptors."""
        en_pairs = []
        mass_pairs = []
        polar_pairs = []
        non_o_total = sum(v for k, v in elements.items() if k != "O")

        for sym, amt in elements.items():
            if sym == "O":
                continue
            props = ELEMENT_REGISTRY.get(sym, {})
            frac = amt / non_o_total if non_o_total > 0 else 0

            en = _safe_float(props.get("en_pauling"))
            if en is not None:
                en_pairs.append((en, frac))

            mass = _safe_float(props.get("atomic_mass"))
            if mass is not None:
                mass_pairs.append((mass, frac))

            pol = _safe_float(props.get("polarizability_a3"))
            if pol is not None:
                polar_pairs.append((pol, frac))

        desc.avg_electronegativity = round(_weighted_mean(en_pairs), 3)
        desc.avg_atomic_mass = round(_weighted_mean(mass_pairs), 2)
        desc.polarizability_index = round(_weighted_mean(polar_pairs), 3)

        # Electronegativity difference: A-site vs B-site
        a_en = []
        for sym, amt in desc.a_site_elements.items():
            en = _safe_float(ELEMENT_REGISTRY.get(sym, {}).get("en_pauling"))
            if en:
                a_en.append((en, amt))
        b_en = []
        for sym, amt in desc.b_site_elements.items():
            en = _safe_float(ELEMENT_REGISTRY.get(sym, {}).get("en_pauling"))
            if en:
                b_en.append((en, amt))
        desc.electronegativity_diff = round(
            abs(_weighted_mean(a_en) - _weighted_mean(b_en)), 3,
        )

    def _classify_crystal_system(self, desc: StructuralDescriptor) -> None:
        """Classify expected crystal system from tolerance factor."""
        t = desc.tolerance_factor
        if t <= 0:
            desc.crystal_system = "undetermined"
            desc.stability_class = "undetermined"
            return

        if 0.99 <= t <= 1.02:
            desc.crystal_system = "cubic"
            desc.stability_class = "highly stable"
        elif 0.96 <= t < 0.99:
            desc.crystal_system = "tetragonal"
            desc.stability_class = "stable (ferroelectric)"
        elif 0.90 <= t < 0.96:
            desc.crystal_system = "orthorhombic"
            desc.stability_class = "stable (tilted octahedra)"
        elif 0.85 <= t < 0.90:
            desc.crystal_system = "rhombohedral"
            desc.stability_class = "marginally stable"
        elif t < 0.85:
            desc.crystal_system = "hexagonal / ilmenite"
            desc.stability_class = "unstable perovskite"
        else:  # t > 1.02
            desc.crystal_system = "hexagonal"
            desc.stability_class = "face-sharing instability"

    def _assess_perovskite(
        self, elements: dict[str, float], desc: StructuralDescriptor,
    ) -> None:
        """Assess likelihood of stable perovskite formation."""
        t = desc.tolerance_factor
        o_f = desc.octahedral_factor
        confidence = 0.0

        # Tolerance factor contribution (ideal: 0.88 – 1.05)
        if 0.88 <= t <= 1.05:
            t_score = 1.0 - abs(t - 0.97) / 0.17
            confidence += t_score * 40
        elif 0.80 <= t < 0.88 or 1.05 < t <= 1.10:
            confidence += 15

        # Octahedral factor contribution (ideal: 0.41 – 0.73)
        if 0.41 <= o_f <= 0.73:
            confidence += 30
        elif 0.35 <= o_f < 0.41 or 0.73 < o_f <= 0.80:
            confidence += 15

        # Has oxygen
        if desc.oxygen_content > 0:
            confidence += 10

        # Has both A and B sites
        if desc.a_site_elements and desc.b_site_elements:
            confidence += 20

        desc.perovskite_confidence = round(min(confidence, 100.0), 1)
        desc.is_perovskite_likely = desc.perovskite_confidence >= 60
