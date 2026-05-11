"""
Piezo.AI — Dashboard Schemas
===============================
Pydantic models for dashboard API requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Stats ──────────────────────────────────────────────────

class SystemStats(BaseModel):
    """System-wide statistics."""
    dataset_count: int = 0
    dataset_ready_count: int = 0
    dataset_pending_count: int = 0
    total_material_rows: int = 0
    trained_model_count: int = 0
    prediction_count: int = 0
    training_job_count: int = 0
    training_completed_count: int = 0
    training_failed_count: int = 0
    db_size_mb: float = 0.0


# ── Models ─────────────────────────────────────────────────

class DashboardModel(BaseModel):
    """Extended trained model info for dashboard."""
    id: str
    display_name: str
    target: str
    algorithm: str
    r2_score: float
    rmse: float
    feature_dim: int
    n_train_samples: int
    n_test_samples: int
    training_duration_s: float
    is_default: bool
    dataset_id: str
    artifact_dir: str
    model_file_path: str
    created_at: datetime


class RenameModelRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=255)


class BulkDeleteModelsRequest(BaseModel):
    model_ids: list[str]


class BulkDeleteModelsResponse(BaseModel):
    deleted_count: int
    errors: list[str] = []


# ── Target Distribution ───────────────────────────────────

class TargetDistribution(BaseModel):
    """Model count per target for donut chart."""
    target: str
    count: int
    percentage: float


# ── Prediction History ────────────────────────────────────

class PredictionHistoryItem(BaseModel):
    """Lightweight prediction record for report selector.
    
    Predictions are grouped by formula+timestamp — member_ids contains
    all individual DB prediction IDs belonging to this group.
    """
    id: str
    member_ids: list[str] = []
    formula: str
    is_composite: bool
    composite_params: Optional[dict] = None
    d33_predicted: Optional[float] = None
    tc_predicted: Optional[float] = None
    hardness_predicted: Optional[float] = None
    prediction_status: str
    created_at: datetime


# ── Report Generation ─────────────────────────────────────

class ReportGenerateRequest(BaseModel):
    """Options for PDF report generation."""
    include_r2_rmse: bool = True
    include_predicted_vs_actual: bool = True
    include_shap_summary: bool = False
    include_ai_insight: bool = False
    include_material_insight: bool = False
    selected_prediction_ids: list[str] = []
    selected_model_ids: list[str] = []


class ReportGenerateResponse(BaseModel):
    """Response after report generation."""
    report_id: str
    filename: str
    download_url: str
    generated_at: datetime
