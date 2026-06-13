"""
Optimization Schemas — Pydantic request/response models.

DUMB PIPE: data validation only, no ML logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- Shared ----------

class OptimizationModelInfo(BaseModel):
    """Trained model available for optimization."""
    id: str
    display_name: str
    target: str
    algorithm: str
    r2_score: float
    rmse: float
    n_train_samples: int
    n_test_samples: int
    feature_dim: int
    is_default: bool


# ---------- Structural Analysis ----------

class StructuralAnalysisRequest(BaseModel):
    formula: str


class StructuralAnalysisCompareRequest(BaseModel):
    formulas: list[str] = Field(..., min_length=1, max_length=10)


class StructuralDescriptorResponse(BaseModel):
    formula: str
    normalized_formula: str
    is_valid: bool = True
    error: str | None = None

    # Goldschmidt
    tolerance_factor: float = 0.0
    octahedral_factor: float = 0.0
    crystal_system: str = "unknown"
    stability_class: str = "unknown"

    # Bond valence
    avg_bond_valence_a: float = 0.0
    avg_bond_valence_b: float = 0.0
    bond_valence_mismatch: float = 0.0

    # Site classification
    a_site_elements: dict[str, float] = {}
    b_site_elements: dict[str, float] = {}
    dopant_elements: dict[str, float] = {}
    oxygen_content: float = 0.0
    total_elements: int = 0

    # Physics descriptors
    avg_electronegativity: float = 0.0
    electronegativity_diff: float = 0.0
    avg_atomic_mass: float = 0.0
    avg_ionic_radius_a: float = 0.0
    avg_ionic_radius_b: float = 0.0
    polarizability_index: float = 0.0
    a_site_variance: float = 0.0
    b_site_variance: float = 0.0

    # Perovskite assessment
    is_perovskite_likely: bool = False
    perovskite_confidence: float = 0.0
    phase_count: int = 1
    warnings: list[str] = []


# ---------- Optimization ----------

class ObjectiveConfig(BaseModel):
    direction: str = "maximize"
    min: float = 0
    max: float = 1000
    weight: float = 1.0


class OptimizationRequest(BaseModel):
    model_ids: dict[str, str]  # target -> model UUID
    objectives: dict[str, ObjectiveConfig] = {}
    preset: str = "custom"
    pop_size: int = Field(default=100, ge=20, le=500)
    n_generations: int = Field(default=50, ge=10, le=300)
    seed: int = 42
    search_elements: list[str] | None = None


class ParetoSolutionResponse(BaseModel):
    composition: dict[str, float]
    formula_approx: str
    predicted: dict[str, float]
    use_case_tag: str = ""
    use_case_color: str = ""
    rank: int = 0
    crowding_distance: float = 0.0


class ConvergencePoint(BaseModel):
    generation: float
    # dynamic keys for each target


class OptimizationResultResponse(BaseModel):
    solutions: list[ParetoSolutionResponse] = []
    convergence: list[dict[str, float]] = []
    n_generations_run: int = 0
    n_evaluations: int = 0
    duration_seconds: float = 0.0
    targets_optimized: list[str] = []
    preset_used: str = "custom"
    error: str | None = None


class UseCasePreset(BaseModel):
    key: str
    label: str
    description: str
    objectives: dict[str, ObjectiveConfig]


class PresetsResponse(BaseModel):
    presets: list[UseCasePreset]
