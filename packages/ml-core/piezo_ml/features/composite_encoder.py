"""
Composite Encoder — shared encoding for composite material features.

Single source of truth for encoding composite columns (matrix_type,
filler_wt_pct, particle_morphology, particle_size_nm, surface_treatment,
fabrication_method, sintering_temp_c, relative_density_pct) into numeric
features. Used by BOTH training pipeline and inference engine.

This ensures training features == inference features for composite params.
"""

from __future__ import annotations

import math
from typing import Any


# Categorical value → integer encodings
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

# The composite feature column names (appended to feature vector)
COMPOSITE_FEATURE_COLUMNS: list[str] = [
    "filler_wt_pct",
    "particle_size_nm",
    "matrix_type_encoded",
    "particle_morphology_encoded",
    "surface_treatment_encoded",
    "fabrication_method_encoded",
    "sintering_temp_c",
    "relative_density_pct",
]


def _safe_float(value: Any) -> float:
    """Convert to float safely, defaulting to 0.0."""
    try:
        v = float(value) if value is not None else 0.0
        return 0.0 if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return 0.0


def encode_composite_params(params: dict[str, Any] | None) -> dict[str, float]:
    """
    Encode composite parameters as numeric features.

    Returns a dict of composite feature columns with numeric values.
    When params is None (bulk ceramic), all values default to 0.
    """
    defaults = {col: 0.0 for col in COMPOSITE_FEATURE_COLUMNS}

    if not params:
        return defaults

    # Direct numeric fields
    defaults["filler_wt_pct"] = _safe_float(params.get("filler_wt_pct", 0))
    defaults["particle_size_nm"] = _safe_float(params.get("particle_size_nm", 0))
    defaults["sintering_temp_c"] = _safe_float(params.get("sintering_temp_c", 0))
    defaults["relative_density_pct"] = _safe_float(params.get("relative_density_pct", 0))

    # Categorical fields → encoded integers
    for field_name, encodings in COMPOSITE_CATEGORICAL_ENCODINGS.items():
        value = str(params.get(field_name, "none")).lower()
        defaults[f"{field_name}_encoded"] = float(encodings.get(value, 0))

    return defaults


def encode_composite_row(row) -> dict[str, float]:
    """
    Encode composite features from a DataFrame row (during training).

    Extracts composite columns from the row and encodes them.
    """
    params: dict[str, Any] = {}
    for field in [
        "matrix_type", "filler_wt_pct", "particle_morphology",
        "particle_size_nm", "surface_treatment", "fabrication_method",
        "sintering_temp_c", "relative_density_pct",
    ]:
        val = row.get(field)
        if val is not None and str(val).lower() not in ("nan", ""):
            params[field] = val
    return encode_composite_params(params)
