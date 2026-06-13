"""
Training module — Pydantic request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TrainingJobCreate(BaseModel):
    """Request body to create a new training job."""
    dataset_id: UUID
    targets: list[str] = Field(..., min_length=1, description="Target columns to train on")
    algorithms: dict[str, str] = Field(..., description="target → algorithm mapping")
    hyperparameters: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="target → {param: value}"
    )
    selected_fields: list[str] = Field(..., min_length=2)
    missing_strategies: dict[str, str] = Field(
        default_factory=dict, description="field → imputation strategy"
    )
    mode: str = Field("manual", pattern="^(manual|auto)$")


class DatasetValidationRequest(BaseModel):
    """Request to pre-validate a dataset before training."""
    dataset_id: UUID
    selected_fields: list[str]
    targets: list[str] = Field(default_factory=list, description="Active training targets")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class FieldIssueResponse(BaseModel):
    field: str
    issue_type: str
    count: int
    total: int
    message: str
    suggestion: str
    default_strategy: str
    allowed_strategies: list[str] = []


class DatasetValidationResponse(BaseModel):
    dataset_id: str
    total_rows: int
    issues: list[FieldIssueResponse]
    default_strategies: dict[str, str]


class TrainingJobResponse(BaseModel):
    id: str
    dataset_id: str
    status: str
    mode: str
    targets: list[str]
    algorithms: dict[str, str]
    hyperparameters: dict[str, Any] | None = None
    selected_fields: list[str]
    progress_pct: float = 0
    current_stage: str | None = None
    initial_rows: int | None = None
    initial_columns: int | None = None
    final_rows: int | None = None
    final_columns: int | None = None
    artifact_dir: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str

    class Config:
        from_attributes = True


class TrainedModelResponse(BaseModel):
    id: str
    display_name: str
    target: str
    algorithm: str
    r2_score: float
    rmse: float
    hyperparameters: dict[str, Any]
    feature_dim: int
    n_train_samples: int
    n_test_samples: int
    model_file_path: str
    training_duration_s: float
    is_default: bool
    created_at: str

    class Config:
        from_attributes = True


class TrainingResultsResponse(BaseModel):
    job_id: str
    status: str
    models: list[TrainedModelResponse]
    convergence_data: dict[str, list[dict[str, float]]] = {}


class HyperparameterDefResponse(BaseModel):
    type: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    options: list[str] | None = None
    description: str = ""
    impact: str = ""
    recommended: Any = None


class AlgorithmInfoResponse(BaseModel):
    key: str
    display_name: str
    description: str
    supports_convergence: bool
    hyperparameters: dict[str, HyperparameterDefResponse]
