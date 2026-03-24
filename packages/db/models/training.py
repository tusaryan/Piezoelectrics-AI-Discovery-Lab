from typing import List, Optional
from datetime import datetime
from sqlalchemy import String, Text, Float, Boolean, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from packages.db.models.base import Base, UUIDMixin

class TrainingJob(Base, UUIDMixin):
    __tablename__ = "training_jobs"
    
    dataset_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    mode: Mapped[str] = mapped_column(String(20)) # auto|compare|expert
    status: Mapped[str] = mapped_column(String(20), default='queued')
    config: Mapped[dict] = mapped_column(JSONB)
    
    train_d33: Mapped[bool] = mapped_column(default=True)
    train_tc: Mapped[bool] = mapped_column(default=True)
    train_hardness: Mapped[bool] = mapped_column(default=False)
    train_composite: Mapped[bool] = mapped_column(default=False)
    
    best_model_d33: Mapped[Optional[str]] = mapped_column(String(50))
    best_model_tc: Mapped[Optional[str]] = mapped_column(String(50))
    r2_d33: Mapped[Optional[float]]
    r2_tc: Mapped[Optional[float]]
    rmse_d33: Mapped[Optional[float]]
    rmse_tc: Mapped[Optional[float]]
    
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255))
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255))
    
    started_at: Mapped[Optional[datetime]]
    completed_at: Mapped[Optional[datetime]]
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    
    logs: Mapped[List["TrainingLog"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    artifacts: Mapped[List["ModelArtifact"]] = relationship(back_populates="job", cascade="all, delete-orphan")

class TrainingLog(Base, UUIDMixin):
    __tablename__ = "training_logs"
    
    job_id: Mapped[UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True)
    job: Mapped["TrainingJob"] = relationship(back_populates="logs")
    
    level: Mapped[str] = mapped_column(String(10)) # INFO, WARN, ERROR
    message: Mapped[str] = mapped_column(Text)
    step: Mapped[Optional[str]] = mapped_column(String(100))
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB)
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)

class ModelArtifact(Base, UUIDMixin):
    __tablename__ = "model_artifacts"
    
    job_id: Mapped[UUID] = mapped_column(ForeignKey("training_jobs.id", ondelete="CASCADE"))
    job: Mapped["TrainingJob"] = relationship(back_populates="artifacts")
    
    target: Mapped[str] = mapped_column(String(20)) # d33, tc, hardness
    model_name: Mapped[str] = mapped_column(String(50)) # base algorithm
    alias: Mapped[Optional[str]] = mapped_column(String(255)) # custom unique name
    is_best: Mapped[bool] = mapped_column(default=False)
    artifact_path: Mapped[str] = mapped_column(String(500))
    mlflow_artifact: Mapped[Optional[str]] = mapped_column(String(500))
    
    r2_train: Mapped[Optional[float]]
    r2_test: Mapped[Optional[float]]
    rmse_train: Mapped[Optional[float]]
    rmse_test: Mapped[Optional[float]]
    mae_test: Mapped[Optional[float]]
    cv_r2_mean: Mapped[Optional[float]]
    cv_r2_std: Mapped[Optional[float]]
    
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSONB)
    shap_values_path: Mapped[Optional[str]] = mapped_column(String(500))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSONB)
    training_time_sec: Mapped[Optional[float]]
    feature_version: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class OptunaStudy(Base, UUIDMixin):
    __tablename__ = "optuna_studies"
    
    job_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("training_jobs.id"))
    model_name: Mapped[str] = mapped_column(String(50))
    target: Mapped[str] = mapped_column(String(20))
    n_trials: Mapped[int]
    completed_trials: Mapped[int] = mapped_column(default=0)
    
    best_r2: Mapped[Optional[float]]
    best_params: Mapped[Optional[dict]] = mapped_column(JSONB)
    all_trials: Mapped[Optional[dict]] = mapped_column(JSONB) # [{number, value, params, duration_sec}]
    param_importances: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    status: Mapped[str] = mapped_column(String(20), default='running')
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[Optional[datetime]]
