"""
Piezo.AI ML Core — models subpackage.

Trainer, algorithm registry, Optuna tuner, model saver,
inference engine, use-case mapper,
and cross-platform multiprocessing utilities.
"""

from piezo_ml.models.platform_utils import configure_multiprocessing, get_safe_n_jobs
from piezo_ml.models.algorithm_registry import (
    ALGORITHM_REGISTRY,
    AlgorithmMeta,
    HyperparamDef,
    build_model,
    get_algorithm_list,
    get_defaults,
)
from piezo_ml.models.trainer import ModelTrainer, TrainingCancelledError, TrainingResult
from piezo_ml.models.optuna_tuner import OptunaTuner
from piezo_ml.models.model_saver import ModelArtifact, save_trained_model
from piezo_ml.models.inference_engine import InferenceEngine, PredictionResult
from piezo_ml.models.use_case_mapper import map_use_case, UseCaseResult, get_use_case_definitions

__all__ = [
    "configure_multiprocessing",
    "get_safe_n_jobs",
    "ALGORITHM_REGISTRY",
    "AlgorithmMeta",
    "HyperparamDef",
    "build_model",
    "get_algorithm_list",
    "get_defaults",
    "ModelTrainer",
    "TrainingCancelledError",
    "TrainingResult",
    "OptunaTuner",
    "ModelArtifact",
    "save_trained_model",
    "InferenceEngine",
    "PredictionResult",
    "map_use_case",
    "UseCaseResult",
    "get_use_case_definitions",
]
