from typing import Optional
from datetime import datetime
from sqlalchemy import String, Text, Float, Integer, ForeignKey, func, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB, UUID
from packages.db.models.base import Base, UUIDMixin

class CompositePrediction(Base, UUIDMixin):
    __tablename__ = "composite_predictions"
    
    matrix_polymer: Mapped[str] = mapped_column(String(100))
    filler_formula: Mapped[str] = mapped_column(String(500))
    volume_fraction: Mapped[float]
    connectivity: Mapped[str] = mapped_column(String(50))
    
    predicted_d33: Mapped[Optional[float]]
    confidence_lower: Mapped[Optional[float]]
    confidence_upper: Mapped[Optional[float]]
    
    properties: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)

class CompositeModelArtifact(Base, UUIDMixin):
    __tablename__ = "composite_model_artifacts"
    
    model_name: Mapped[str] = mapped_column(String(255))
    target: Mapped[str] = mapped_column(String(50), default='composite_d33')
    r2_test: Mapped[Optional[float]]
    rmse_test: Mapped[Optional[float]]
    artifact_path: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=False)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
