"""
Interpret Module — Pydantic schemas for SHAP, Physics Validation, and PySR.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- Request Schemas ----------

class ShapBeeswarmRequest(BaseModel):
    """Request to compute SHAP beeswarm (global importance)."""
    model_id: str = Field(..., description="UUID of trained model")
    max_samples: int = Field(200, ge=10, le=1000, description="Max samples for SHAP computation")


class ShapWaterfallRequest(BaseModel):
    """Request to compute SHAP waterfall (single sample)."""
    model_id: str = Field(..., description="UUID of trained model")
    sample_index: int = Field(0, ge=0, description="Sample index in training data")


class ShapDependenceRequest(BaseModel):
    """Request to compute SHAP dependence for a feature."""
    model_id: str = Field(..., description="UUID of trained model")
    feature_name: str = Field(..., description="Feature name to analyze")


class PhysicsValidationRequest(BaseModel):
    """Request to run physics validation on SHAP data."""
    model_id: str = Field(..., description="UUID of trained model")


class SymbolicRegressionRequest(BaseModel):
    """Request to run PySR symbolic regression."""
    model_id: str = Field(..., description="UUID of trained model")
    max_complexity: int = Field(20, ge=5, le=40)
    n_iterations: int = Field(40, ge=10, le=200)
    timeout_seconds: int = Field(120, ge=30, le=600)


# ---------- Response Schemas ----------

class BeeswarmFeature(BaseModel):
    """Single feature in beeswarm response."""
    name: str
    mean_abs_shap: float
    rank: int


class ShapBeeswarmResponse(BaseModel):
    """Response with SHAP beeswarm data."""
    model_id: str
    target: str
    algorithm: str
    feature_names: list[str]
    shap_values: list[list[float]]
    feature_values: list[list[float]]
    base_value: float
    mean_abs_shap: list[float]
    top_features: list[BeeswarmFeature]
    n_samples: int


class ShapWaterfallResponse(BaseModel):
    """Response with SHAP waterfall data."""
    model_id: str
    target: str
    feature_names: list[str]
    shap_values: list[float]
    feature_values: list[float]
    base_value: float
    prediction: float
    sample_index: int
    n_total_samples: int


class ShapDependenceResponse(BaseModel):
    """Response with SHAP dependence data."""
    model_id: str
    target: str
    feature_name: str
    feature_values: list[float]
    shap_values: list[float]
    interaction_feature: str | None
    interaction_values: list[float]


class PhysicsCheckItem(BaseModel):
    """Single physics validation check."""
    feature: str
    expected_effect: str
    physics_reason: str
    actual_effect: str
    aligned: bool
    shap_magnitude: float
    shap_rank: int


class PhysicsValidationResponse(BaseModel):
    """Response with physics validation results."""
    model_id: str
    target: str
    alignment_score: float
    total_checks: int
    confirmed: int
    violations: list[PhysicsCheckItem]
    confirmed_checks: list[PhysicsCheckItem]
    skipped: list[str]


class EquationItem(BaseModel):
    """Single equation from symbolic regression."""
    equation_str: str
    latex: str
    complexity: int
    loss: float
    r2: float
    readable: str


class ParetoPoint(BaseModel):
    """Point on the parsimony Pareto front."""
    complexity: float
    loss: float
    r2: float


class SymbolicRegressionResponse(BaseModel):
    """Response with PySR results."""
    model_id: str
    target: str
    equations: list[EquationItem]
    best_equation: EquationItem | None
    pareto_front: list[ParetoPoint]
    n_samples: int
    n_features: int
    available: bool
    error: str | None = None


class InterpretModelInfo(BaseModel):
    """Model info for the interpret model selector."""
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
