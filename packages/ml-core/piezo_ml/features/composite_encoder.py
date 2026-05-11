from typing import Any

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

def encode_composite_params(params: dict[str, Any] | None) -> dict[str, float]:
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

    defaults["filler_wt_pct"] = float(params.get("filler_wt_pct", 0) or 0)
    defaults["particle_size_nm"] = float(params.get("particle_size_nm", 0) or 0)
    defaults["sintering_temp_c"] = float(params.get("sintering_temp_c", 0) or 0)
    defaults["relative_density_pct"] = float(params.get("relative_density_pct", 0) or 0)

    for field_name, encodings in COMPOSITE_CATEGORICAL_ENCODINGS.items():
        value = str(params.get(field_name, "none")).lower()
        defaults[f"{field_name}_encoded"] = float(encodings.get(value, 0))

    return defaults
