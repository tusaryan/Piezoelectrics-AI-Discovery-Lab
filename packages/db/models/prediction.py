from typing import Optional
from datetime import datetime
from sqlalchemy import String, Text, Float, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from packages.db.models.base import Base, UUIDMixin

class Prediction(Base, UUIDMixin):
    __tablename__ = "predictions"
    
    formula: Mapped[str] = mapped_column(String(500))
    sintering_temp: Mapped[Optional[float]]
    composite_config: Mapped[Optional[dict]] = mapped_column(JSONB)
    model_artifact_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("model_artifacts.id"))
    
    predicted_d33: Mapped[Optional[float]]
    predicted_tc: Mapped[Optional[float]]
    predicted_hardness: Mapped[Optional[float]]
    predicted_composite_d33: Mapped[Optional[float]]
    
    d33_lower_95: Mapped[Optional[float]]
    d33_upper_95: Mapped[Optional[float]]
    tc_lower_95: Mapped[Optional[float]]
    tc_upper_95: Mapped[Optional[float]]
    
    parsed_features: Mapped[Optional[dict]] = mapped_column(JSONB)
    session_id: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)

class ParetoRun(Base, UUIDMixin):
    __tablename__ = "pareto_runs"
    
    name: Mapped[str] = mapped_column(String(255))
    algorithm: Mapped[str] = mapped_column(String(20)) # NSGA-II|MOBO
    objectives: Mapped[dict] = mapped_column(JSONB)
    constraints: Mapped[Optional[dict]] = mapped_column(JSONB)
    n_generations: Mapped[int] = mapped_column(default=100)
    population_size: Mapped[int] = mapped_column(default=200)
    status: Mapped[str] = mapped_column(String(20), default='running')
    result_count: Mapped[Optional[int]]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class ParetoSolution(Base, UUIDMixin):
    __tablename__ = "pareto_solutions"
    
    run_id: Mapped[UUID] = mapped_column(ForeignKey("pareto_runs.id", ondelete="CASCADE"), index=True)
    formula: Mapped[Optional[str]] = mapped_column(String(500))
    composition: Mapped[dict] = mapped_column(JSONB)
    
    predicted_d33: Mapped[Optional[float]]
    predicted_tc: Mapped[Optional[float]]
    predicted_hardness: Mapped[Optional[float]]
    
    rank: Mapped[Optional[int]]
    crowd_distance: Mapped[Optional[float]]
    use_case: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class SymbolicRegressionRun(Base, UUIDMixin):
    __tablename__ = "symbolic_regression_runs"
    
    target: Mapped[str] = mapped_column(String(20))
    algorithm: Mapped[str] = mapped_column(String(20), default='PySR')
    discovered_equations: Mapped[Optional[dict]] = mapped_column(JSONB)
    best_equation: Mapped[Optional[str]] = mapped_column(Text)
    best_r2: Mapped[Optional[float]]
    runtime_sec: Mapped[Optional[float]]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class ActiveLearningRun(Base, UUIDMixin):
    __tablename__ = "active_learning_runs"
    
    strategy: Mapped[str] = mapped_column(String(30))
    acquisition_fn: Mapped[Optional[str]] = mapped_column(String(30))
    total_budget: Mapped[int]
    current_iteration: Mapped[int] = mapped_column(default=0)
    found_max_d33_at_iter: Mapped[Optional[int]]
    final_max_d33: Mapped[Optional[float]]
    efficiency_vs_random: Mapped[Optional[float]]
    status: Mapped[str] = mapped_column(String(20), default='running')
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class AgentConversation(Base, UUIDMixin):
    __tablename__ = "agent_conversations"
    
    session_id: Mapped[str] = mapped_column(String(255))
    title: Mapped[Optional[str]] = mapped_column(String(255))
    messages: Mapped[dict] = mapped_column(JSONB, default=list)
    model_used: Mapped[str] = mapped_column(String(50), default='claude-sonnet-4-5')
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

class SystemConfig(Base):
    __tablename__ = "system_config"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
