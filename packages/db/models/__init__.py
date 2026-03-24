from packages.db.models.base import Base, TimestampMixin, UUIDMixin
from packages.db.models.dataset import Dataset, Material, ElementFraction, AtomicDescriptors
from packages.db.models.training import TrainingJob, TrainingLog, ModelArtifact, OptunaStudy
from packages.db.models.prediction import (
    Prediction, ParetoRun, ParetoSolution, 
    SymbolicRegressionRun, ActiveLearningRun, 
    AgentConversation, SystemConfig
)
from packages.db.models.composite import CompositePrediction, CompositeModelArtifact

# Export all models for Alembic base.metadata
__all__ = [
    "Base", 
    "Dataset", "Material", "ElementFraction", "AtomicDescriptors",
    "TrainingJob", "TrainingLog", "ModelArtifact", "OptunaStudy",
    "Prediction", "ParetoRun", "ParetoSolution",
    "SymbolicRegressionRun", "ActiveLearningRun",
    "AgentConversation", "SystemConfig",
    "CompositePrediction", "CompositeModelArtifact"
]
