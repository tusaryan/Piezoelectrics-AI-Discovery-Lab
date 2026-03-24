from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, Float, Boolean, Integer, ForeignKey, SmallInteger, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from packages.db.models.base import Base, TimestampMixin, UUIDMixin

class Dataset(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "datasets"
    
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[int] = mapped_column(default=1)
    is_active: Mapped[bool] = mapped_column(default=True)
    source: Mapped[str] = mapped_column(String(50), default='upload')
    status: Mapped[str] = mapped_column(String(50), default='ready')
    row_count: Mapped[Optional[int]]
    has_d33: Mapped[bool] = mapped_column(default=False)
    has_tc: Mapped[bool] = mapped_column(default=False)
    column_map: Mapped[Optional[dict]] = mapped_column(JSONB)
    upload_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    materials: Mapped[List["Material"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")

class Material(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "materials"
    
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    dataset: Mapped["Dataset"] = relationship(back_populates="materials")
    
    # CORE
    formula: Mapped[str] = mapped_column(String(500), index=True)
    d33: Mapped[Optional[float]]
    tc: Mapped[Optional[float]]
    is_tc_ai_generated: Mapped[bool] = mapped_column(default=False)
    
    # EXTENDED KNN BULK
    family_name: Mapped[Optional[str]] = mapped_column(String(100))
    sintering_temp: Mapped[Optional[float]]
    field_strength: Mapped[Optional[float]]
    poling_temp: Mapped[Optional[float]]
    poling_time: Mapped[Optional[float]]
    density: Mapped[Optional[float]]
    density_theoretical_pct: Mapped[Optional[float]]
    planar_coupling: Mapped[Optional[float]]
    dielectric_const: Mapped[Optional[float]]
    dielectric_loss: Mapped[Optional[float]]
    mech_quality_factor: Mapped[Optional[float]]
    ot_transition: Mapped[Optional[float]]
    lattice_ratio: Mapped[Optional[float]]
    max_strain: Mapped[Optional[float]]
    
    # HARDNESS
    vickers_hardness: Mapped[Optional[float]]
    mohs_hardness: Mapped[Optional[float]]
    
    # COMPOSITE
    is_composite: Mapped[bool] = mapped_column(default=False, index=True)
    matrix_type: Mapped[Optional[str]] = mapped_column(String(50))
    filler_wt_pct: Mapped[Optional[float]]
    filler_vol_pct: Mapped[Optional[float]]
    particle_morphology: Mapped[Optional[str]] = mapped_column(String(50))
    particle_size_nm: Mapped[Optional[float]]
    surface_treatment: Mapped[Optional[str]] = mapped_column(String(100))
    fabrication_method: Mapped[Optional[str]] = mapped_column(String(100))
    beta_phase_pct: Mapped[Optional[float]]
    composite_d33: Mapped[Optional[float]]
    remnant_polarization: Mapped[Optional[float]]
    coercive_field: Mapped[Optional[float]]
    
    # METADATA
    reference_doi: Mapped[Optional[str]] = mapped_column(String(255))
    reference_num: Mapped[Optional[str]] = mapped_column(String(50))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    data_quality: Mapped[int] = mapped_column(SmallInteger, default=3)
    is_verified: Mapped[bool] = mapped_column(default=False)
    is_imputed: Mapped[bool] = mapped_column(default=False)
    
    fractions: Mapped[List["ElementFraction"]] = relationship(back_populates="material", cascade="all, delete-orphan")
    descriptors: Mapped[Optional["AtomicDescriptors"]] = relationship(back_populates="material", uselist=False, cascade="all, delete-orphan")

class ElementFraction(Base, UUIDMixin):
    __tablename__ = "element_fractions"
    
    material_id: Mapped[UUID] = mapped_column(ForeignKey("materials.id", ondelete="CASCADE"))
    material: Mapped["Material"] = relationship(back_populates="fractions")
    
    element: Mapped[str] = mapped_column(String(5))
    fraction: Mapped[float]
    site: Mapped[Optional[str]] = mapped_column(String(10))

class AtomicDescriptors(Base, UUIDMixin):
    __tablename__ = "atomic_descriptors"
    
    material_id: Mapped[UUID] = mapped_column(ForeignKey("materials.id", ondelete="CASCADE"), unique=True)
    material: Mapped["Material"] = relationship(back_populates="descriptors")
    
    descriptors: Mapped[dict] = mapped_column(JSONB)
    feature_version: Mapped[str] = mapped_column(String(20), default='v2')
    computed_at: Mapped[Optional[datetime]] = mapped_column(server_default=func.now())
