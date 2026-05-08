from __future__ import annotations

TARGET_FIELDS: tuple[str, ...] = ("d33", "tc", "vickers_hardness")
BASE_INPUT_FIELDS: tuple[str, ...] = (
    "formula",
    "qm",
    "kp",
    "relative_density_pct",
    "sintering_temp_c",
    "sintering_method",
    "ceramic_type",
    "fabrication_method",
    "matrix_type",
    "filler_wt_pct",
    "particle_morphology",
    "particle_size_nm",
    "surface_treatment",
)

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "sintering_method",
    "ceramic_type",
    "fabrication_method",
    "matrix_type",
    "particle_morphology",
    "surface_treatment",
)

NUMERIC_FIELDS: tuple[str, ...] = tuple(
    f for f in BASE_INPUT_FIELDS if f not in CATEGORICAL_FIELDS and f != "formula"
)


def get_trainable_targets(columns: list[str]) -> list[str]:
    return [target for target in TARGET_FIELDS if target in columns]


def get_feature_candidates(columns: list[str]) -> list[str]:
    return [field for field in BASE_INPUT_FIELDS if field in columns]
