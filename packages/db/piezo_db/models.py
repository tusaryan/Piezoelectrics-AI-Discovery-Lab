"""
Piezo.AI — Database Models
============================
SQLAlchemy ORM models matching the schema defined in
02-cross-cutting-and-build-plan.md §5.6.

IMPORTANT: This file is the SINGLE SOURCE OF TRUTH for DB table structure.
Any schema changes must be reflected here AND in the plan document.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from piezo_db.base import Base


# ---------------------------------------------------------------------------
# datasets — Uploaded CSV metadata
# ---------------------------------------------------------------------------
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    display_name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending | ready
    total_rows = Column(Integer, nullable=False)
    total_columns = Column(Integer, nullable=False)
    column_mapping = Column(JSONB, nullable=False)  # {"original": "backend_field"}
    has_composite_fields = Column(Boolean, nullable=False, default=False)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    materials = relationship("Material", back_populates="dataset", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="dataset", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_datasets_status", "status"),
    )


# ---------------------------------------------------------------------------
# materials — Individual rows of a dataset (post-column-mapping)
# ---------------------------------------------------------------------------
class Material(Base):
    __tablename__ = "materials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    uid = Column(Integer, nullable=False)  # Sequential from 1, per dataset. NOT DB row index.

    # Core fields
    formula = Column(String(500), nullable=False)
    d33 = Column(Float, nullable=True)  # pC/N. Ceramic: 66–680. Composite: 0–78
    tc = Column(Float, nullable=True)  # °C. Range: 105–458
    vickers_hardness = Column(Float, nullable=True)  # HV kgf/mm². Range: 38–1200
    qm = Column(Float, nullable=True)  # Dimensionless. Range: 50–1500+
    kp = Column(Float, nullable=True)  # Dimensionless. Range: 0.30–0.65
    relative_density_pct = Column(Float, nullable=True)  # %. Range: 88–99
    sintering_temp_c = Column(Float, nullable=True)  # °C. Range: 850–1160

    # Categorical fields
    sintering_method = Column(String(50), nullable=True)
    # Valid: conventional|hot_press|sps|rtgg|tgg|two_step|cold_sinter
    ceramic_type = Column(String(20), nullable=True)
    # Valid: soft|hard|composite
    fabrication_method = Column(String(50), nullable=True)
    # Valid: conventional|hot_press|sps|rtgg|tgg|two_step|electrospinning|solvent_cast|cold_sinter|hot_compression

    # Composite fields (sentinel defaults for bulk ceramics)
    matrix_type = Column(String(50), nullable=False, default="none")
    # Valid: none|pvdf|p_vdf_trfe|pvdf_hfp|pvdf_hfp_ctrfe
    filler_wt_pct = Column(Float, nullable=False, default=0)
    # 0 for bulk. Composite: 3,5,10,15,20,40,80
    particle_morphology = Column(String(30), nullable=False, default="none")
    # Valid: none|spherical|rod|cube|nanoblock|fiber|platelet
    particle_size_nm = Column(Float, nullable=True)
    # NULL for bulk. Composite: 50–1000
    surface_treatment = Column(String(30), nullable=False, default="none")
    # Valid: none|untreated|silane|plasma|acid|peg|dopamine

    # Traceability
    source_doi = Column(String(500), nullable=True)
    source_notes = Column(Text, nullable=True)

    # Parse status
    parse_status = Column(String(30), nullable=False, default="pending")
    # Valid: pending|success|error|unsupported_elements
    parse_warnings = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="materials")

    __table_args__ = (
        UniqueConstraint("dataset_id", "uid", name="uq_materials_dataset_uid"),
        Index("ix_materials_dataset_id", "dataset_id"),
        Index("ix_materials_formula", "formula"),
    )


# ---------------------------------------------------------------------------
# training_jobs — Training pipeline execution state
# ---------------------------------------------------------------------------
class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    # Valid: queued|running|completed|failed|cancelled
    mode = Column(String(10), nullable=False, default="manual")
    # Valid: manual|auto
    targets = Column(JSONB, nullable=False)  # ["d33", "tc", "vickers_hardness"]
    algorithms = Column(JSONB, nullable=False)  # {"d33": "xgboost", "tc": "random_forest"}
    hyperparameters = Column(JSONB, nullable=True)
    selected_fields = Column(JSONB, nullable=False)  # ["formula", "d33", "tc", ...]
    progress_pct = Column(Float, nullable=False, default=0)  # 0.0–100.0
    current_stage = Column(String(100), nullable=True)  # Shown above progress bar
    initial_rows = Column(Integer, nullable=True)
    initial_columns = Column(Integer, nullable=True)
    final_rows = Column(Integer, nullable=True)
    final_columns = Column(Integer, nullable=True)
    artifact_dir = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    trained_models = relationship("TrainedModel", back_populates="training_job", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_training_jobs_status", "status"),
        Index("ix_training_jobs_dataset_id", "dataset_id"),
    )


# ---------------------------------------------------------------------------
# trained_models — Trained model metadata (one row per target per training job)
# ---------------------------------------------------------------------------
class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    display_name = Column(String(255), nullable=False)  # User-renameable
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    target = Column(String(30), nullable=False)  # d33|tc|vickers_hardness
    algorithm = Column(String(30), nullable=False)
    # Valid: xgboost|random_forest|svr|lightgbm|gradient_boosting|decision_tree|ann|stacking
    r2_score = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    hyperparameters = Column(JSONB, nullable=False)
    feature_version = Column(String(10), nullable=False)  # e.g., "v4"
    feature_dim = Column(Integer, nullable=False)
    n_train_samples = Column(Integer, nullable=False)
    n_test_samples = Column(Integer, nullable=False)
    supported_elements = Column(JSONB, nullable=False)  # ["K", "Na", "Nb", "O", ...]
    model_file_path = Column(String(500), nullable=False)
    artifact_dir = Column(String(500), nullable=False)
    training_duration_s = Column(Float, nullable=False)
    is_default = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="trained_models")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_trained_models_target", "target"),
        Index("ix_trained_models_dataset_id", "dataset_id"),
        Index("ix_trained_models_is_default", "target", "is_default"),
    )


# ---------------------------------------------------------------------------
# predictions — Individual prediction history
# ---------------------------------------------------------------------------
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("trained_models.id"), nullable=False)
    batch_id = Column(UUID(as_uuid=True), ForeignKey("prediction_batches.id"), nullable=True)
    formula = Column(String(500), nullable=False)
    is_composite = Column(Boolean, nullable=False, default=False)
    composite_params = Column(JSONB, nullable=True)
    d33_predicted = Column(Float, nullable=True)
    d33_ci_lower = Column(Float, nullable=True)
    d33_ci_upper = Column(Float, nullable=True)
    tc_predicted = Column(Float, nullable=True)
    tc_ci_lower = Column(Float, nullable=True)
    tc_ci_upper = Column(Float, nullable=True)
    hardness_predicted = Column(Float, nullable=True)
    prediction_status = Column(String(30), nullable=False)
    # Valid: success|unsupported_elements|parse_error
    prediction_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    model = relationship("TrainedModel", back_populates="predictions")
    batch = relationship("PredictionBatch", back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_model_id", "model_id"),
        Index("ix_predictions_batch_id", "batch_id"),
        Index("ix_predictions_formula", "formula"),
    )


# ---------------------------------------------------------------------------
# prediction_batches — Batch prediction job metadata
# ---------------------------------------------------------------------------
class PredictionBatch(Base):
    __tablename__ = "prediction_batches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("trained_models.id"), nullable=False)
    source_filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, nullable=False)
    success_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    result_file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    predictions = relationship("Prediction", back_populates="batch")

    __table_args__ = (
        Index("ix_prediction_batches_model_id", "model_id"),
    )
