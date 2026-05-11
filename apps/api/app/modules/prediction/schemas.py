"""
Prediction module — Pydantic request/response schemas.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for a single prediction."""
    formula: str = Field(..., min_length=1, description="Chemical formula")
    model_id: UUID = Field(..., description="Trained model UUID")
    composite_params: dict[str, Any] | None = Field(
        None, description="Composite material parameters"
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction from existing dataset."""
    model_ids: dict[str, str] = Field(..., description="Per-target model IDs: {d33: uuid, tc: uuid, vickers_hardness: uuid}")
    dataset_id: UUID


class FormulaValidateRequest(BaseModel):
    """Request for real-time formula validation."""
    formula: str
    strict_mode: bool = False


class ModelRenameRequest(BaseModel):
    """Request to rename a trained model."""
    display_name: str = Field(..., min_length=1, max_length=255)


class ModelSetDefaultRequest(BaseModel):
    """Request to set model as default for its target."""
    is_default: bool = True


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PropertyPrediction(BaseModel):
    """Predicted property with confidence interval."""
    value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


class UseCaseInfo(BaseModel):
    """Predicted use-case for a material."""
    name: str
    category: str
    confidence: float
    description: str
    icon: str
    color: str
    tier: str | None = None
    tier_label: str | None = None
    driving_properties: list[str] | None = None
    score: int | None = None


class UsagePredictionsInfo(BaseModel):
    """Full usage prediction data with multiple recommendations."""
    recommendations: list[UseCaseInfo] = []
    caution_notes: list[str] = []
    property_completeness: str = "full"
    properties_used: list[str] = []


class PredictResponse(BaseModel):
    """Response for a single prediction."""
    formula: str
    is_composite: bool
    status: str  # success | unsupported_elements | parse_error
    notes: str | None = None
    d33: PropertyPrediction | None = None
    tc: PropertyPrediction | None = None
    hardness: PropertyPrediction | None = None
    use_case: UseCaseInfo | None = None
    usage_predictions: UsagePredictionsInfo | None = None
    composite_params: dict[str, Any] | None = None


class BatchResultRow(BaseModel):
    """A single row from batch prediction results — for tabular preview."""
    uid: int | None = None
    formula: str
    is_composite: bool = False
    d33_predicted: float | None = None
    d33_ci_lower: float | None = None
    d33_ci_upper: float | None = None
    tc_predicted: float | None = None
    tc_ci_lower: float | None = None
    tc_ci_upper: float | None = None
    hardness_predicted: float | None = None
    hardness_ci_lower: float | None = None
    hardness_ci_upper: float | None = None
    top_use_case: str | None = None
    use_case_score: int | None = None
    prediction_status: str = "success"
    prediction_notes: str | None = None


class BatchPredictSummary(BaseModel):
    """Summary of a batch prediction."""
    batch_id: str
    total_rows: int
    success_count: int
    error_count: int
    result_file_path: str | None = None
    source_filename: str
    results: list[BatchResultRow] = []


class FormulaValidateResponse(BaseModel):
    """Response for formula validation."""
    formula: str
    is_valid: bool
    normalized_formula: str | None = None
    elements: dict[str, float] | None = None
    unsupported: list[str] | None = None
    error: str | None = None
    warnings: list[str] = []


class TrainedModelListItem(BaseModel):
    """Model summary for model selector dropdown."""
    id: str
    display_name: str
    target: str
    algorithm: str
    r2_score: float
    rmse: float
    feature_dim: int
    n_train_samples: int
    n_test_samples: int
    supported_elements: list[str]
    is_default: bool
    created_at: str

    class Config:
        from_attributes = True


class SupportedElementsResponse(BaseModel):
    """Response listing all supported elements."""
    elements: list[str]
    count: int
