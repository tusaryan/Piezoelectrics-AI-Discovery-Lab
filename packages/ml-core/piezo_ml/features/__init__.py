"""Piezo.AI ML Core feature engineering."""

from piezo_ml.features.feature_engineer import EngineeredRow, FeatureEngineer
from piezo_ml.features.field_registry import (
    BASE_INPUT_FIELDS,
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    TARGET_FIELDS,
    get_feature_candidates,
    get_trainable_targets,
)
from piezo_ml.features.composite_encoder import (
    COMPOSITE_CATEGORICAL_ENCODINGS,
    COMPOSITE_FEATURE_COLUMNS,
    encode_composite_params,
    encode_composite_row,
)

__all__ = [
    "EngineeredRow",
    "FeatureEngineer",
    "TARGET_FIELDS",
    "BASE_INPUT_FIELDS",
    "CATEGORICAL_FIELDS",
    "NUMERIC_FIELDS",
    "get_trainable_targets",
    "get_feature_candidates",
    "COMPOSITE_CATEGORICAL_ENCODINGS",
    "COMPOSITE_FEATURE_COLUMNS",
    "encode_composite_params",
    "encode_composite_row",
]

