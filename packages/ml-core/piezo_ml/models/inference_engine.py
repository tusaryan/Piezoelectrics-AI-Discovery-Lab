"""
Inference Engine — load trained models and run predictions.

Loads .joblib model + metadata JSON, engineers features from formula,
and returns predicted properties with confidence intervals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from piezo_ml.features import FeatureEngineer
from piezo_ml.parsers import FormulaParser
from piezo_ml.registry import get_supported_elements_list, get_unsupported_elements


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    formula: str
    is_composite: bool
    status: str  # "success" | "unsupported_elements" | "parse_error"
    notes: str | None = None

    # Predicted values (None if not predicted by model)
    d33: float | None = None
    d33_ci_lower: float | None = None
    d33_ci_upper: float | None = None
    tc: float | None = None
    tc_ci_lower: float | None = None
    tc_ci_upper: float | None = None
    hardness: float | None = None

    # Composite params used
    composite_params: dict[str, Any] | None = None

    # Use-case mapping
    use_case: dict[str, Any] | None = None


@dataclass
class LoadedModel:
    """A loaded model with its metadata."""
    model: Any
    metadata: dict[str, Any]
    target: str
    algorithm: str
    feature_dim: int
    supported_elements: list[str]
    model_path: str


COMPOSITE_CATEGORICAL_ENCODINGS: dict[str, dict[str, int]] = {
    "matrix_type": {
        "none": 0, "pvdf": 1, "p_vdf_trfe": 2,
        "pvdf_hfp": 3, "pvdf_hfp_ctrfe": 4,
        "epoxy": 5, "silicone": 6, "pvdf_trfe": 7,
    },
    "particle_morphology": {
        "none": 0, "spherical": 1, "rod": 2, "cube": 3,
        "nanoblock": 4, "fiber": 5, "platelet": 6,
    },
    "surface_treatment": {
        "none": 0, "untreated": 1, "silane": 2, "plasma": 3,
        "acid": 4, "peg": 5, "dopamine": 6, "fluorinated": 7,
    },
    "fabrication_method": {
        "conventional": 0, "hot_press": 1, "sps": 2, "rtgg": 3,
        "tgg": 4, "two_step": 5, "electrospinning": 6,
        "solvent_cast": 7, "cold_sinter": 8, "hot_compression": 9,
    },
}


class InferenceEngine:
    """Loads trained models and runs predictions."""

    def __init__(self) -> None:
        self.parser = FormulaParser()
        self.engineer = FeatureEngineer()
        self._model_cache: dict[str, LoadedModel] = {}

    def load_model(self, model_file_path: str, metadata: dict[str, Any]) -> LoadedModel:
        """Load a model from disk (with in-memory caching)."""
        cache_key = model_file_path
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_path = Path(model_file_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[4] / model_file_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        loaded = LoadedModel(
            model=model,
            metadata=metadata,
            target=metadata.get("targets", ["d33"])[0],
            algorithm=list(metadata.get("algorithms", {"d33": "unknown"}).values())[0],
            feature_dim=metadata.get("feature_dim", 0),
            supported_elements=metadata.get("supported_elements", []),
            model_path=str(model_path),
        )
        self._model_cache[cache_key] = loaded
        return loaded

    def predict_single(
        self,
        formula: str,
        model_file_path: str,
        metadata: dict[str, Any],
        composite_params: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Run prediction for a single formula."""
        # Parse and validate formula
        parsed = self.parser.parse(formula)
        if not parsed.is_valid:
            if parsed.unsupported:
                return PredictionResult(
                    formula=formula,
                    is_composite=False,
                    status="unsupported_elements",
                    notes=f"Unsupported elements: {', '.join(parsed.unsupported)}",
                )
            return PredictionResult(
                formula=formula,
                is_composite=False,
                status="parse_error",
                notes=parsed.error or "Could not parse formula",
            )

        # Detect bulk vs composite
        is_composite = _detect_composite(composite_params)

        # Load model
        loaded = self.load_model(model_file_path, metadata)

        # Engineer features
        try:
            engineered = self.engineer.engineer_row(uid=1, formula=formula)
        except ValueError as e:
            return PredictionResult(
                formula=formula, is_composite=is_composite,
                status="parse_error", notes=str(e),
            )

        # Build feature vector
        feature_vector = self._build_feature_vector(
            engineered, composite_params, loaded,
        )

        # Run prediction
        try:
            prediction = loaded.model.predict(feature_vector)[0]
        except Exception as e:
            return PredictionResult(
                formula=formula, is_composite=is_composite,
                status="parse_error", notes=f"Prediction failed: {e}",
            )

        # Compute confidence interval
        ci_lower, ci_upper = self._compute_ci(loaded, feature_vector, prediction)

        result = PredictionResult(
            formula=formula,
            is_composite=is_composite,
            status="success",
            composite_params=composite_params if is_composite else None,
        )

        target = loaded.target
        if target == "d33":
            result.d33 = round(float(prediction), 2)
            result.d33_ci_lower = round(ci_lower, 2)
            result.d33_ci_upper = round(ci_upper, 2)
        elif target == "tc":
            result.tc = round(float(prediction), 2)
            result.tc_ci_lower = round(ci_lower, 2)
            result.tc_ci_upper = round(ci_upper, 2)
        elif target == "vickers_hardness":
            result.hardness = round(float(prediction), 2)

        return result

    def _build_feature_vector(
        self,
        engineered: Any,
        composite_params: dict[str, Any] | None,
        loaded: LoadedModel,
    ) -> np.ndarray:
        """Build feature vector matching training feature columns."""
        # Base features: elemental fractions + weighted properties
        base = {}
        base.update({f"frac_{k}": v for k, v in engineered.element_fractions.items()})
        base.update(engineered.weighted_features)

        # Composite features
        comp = _encode_composite_params(composite_params)
        base.update(comp)

        # Align to training feature columns from metadata
        feature_cols = loaded.metadata.get("feature_columns", [])
        if not feature_cols:
            # Fallback: use whatever we have, padded to feature_dim
            vec = list(base.values())
            while len(vec) < loaded.feature_dim:
                vec.append(0.0)
            return np.array([vec[:loaded.feature_dim]])

        row = {col: base.get(col, 0.0) for col in feature_cols}
        df = pd.DataFrame([row])
        return df.values

    def _compute_ci(
        self, loaded: LoadedModel, features: np.ndarray, prediction: float,
    ) -> tuple[float, float]:
        """Compute 95% confidence interval."""
        model = loaded.model

        # For tree ensembles: use individual tree predictions
        if hasattr(model, "estimators_"):
            try:
                if hasattr(model, "n_estimators"):
                    preds = np.array([
                        est.predict(features)[0]
                        for est in model.estimators_
                    ])
                    std = float(np.std(preds))
                    return prediction - 1.96 * std, prediction + 1.96 * std
            except Exception:
                pass

        # Fallback: ±10% of prediction magnitude
        margin = abs(prediction) * 0.10
        return prediction - margin, prediction + margin

    def clear_cache(self) -> None:
        """Clear model cache."""
        self._model_cache.clear()


def _detect_composite(params: dict[str, Any] | None) -> bool:
    """Detect if prediction is for a composite material."""
    if not params:
        return False
    filler = params.get("filler_wt_pct", 0)
    matrix = params.get("matrix_type", "none")
    try:
        filler_val = float(filler)
    except (TypeError, ValueError):
        filler_val = 0
    return filler_val > 0 and str(matrix).lower() != "none"


def _encode_composite_params(params: dict[str, Any] | None) -> dict[str, float]:
    """Encode composite parameters as numeric features."""
    defaults = {
        "filler_wt_pct": 0.0,
        "particle_size_nm": 0.0,
        "matrix_type_encoded": 0.0,
        "particle_morphology_encoded": 0.0,
        "surface_treatment_encoded": 0.0,
        "fabrication_method_encoded": 0.0,
        "sintering_temp_c": 0.0,
        "relative_density_pct": 0.0,
    }
    if not params:
        return defaults

    defaults["filler_wt_pct"] = float(params.get("filler_wt_pct", 0))
    defaults["particle_size_nm"] = float(params.get("particle_size_nm", 0) or 0)
    defaults["sintering_temp_c"] = float(params.get("sintering_temp_c", 0) or 0)
    defaults["relative_density_pct"] = float(params.get("relative_density_pct", 0) or 0)

    for field_name, encodings in COMPOSITE_CATEGORICAL_ENCODINGS.items():
        value = str(params.get(field_name, "none")).lower()
        defaults[f"{field_name}_encoded"] = float(encodings.get(value, 0))

    return defaults
