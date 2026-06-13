"""
Model Saver — persist trained models as .joblib files with metadata JSON.

Saves to resources/trained-models/ with timestamp-based naming.
Metadata links model to dataset, training config, and parsed artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelArtifact:
    """Result of saving a model to disk."""
    model_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def _get_default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "resources" / "trained-models"


def save_trained_model(
    model: Any,
    target: str,
    algorithm: str,
    metrics: dict[str, float],
    hyperparameters: dict[str, Any],
    feature_dim: int,
    n_train: int,
    n_test: int,
    supported_elements: list[str],
    dataset_id: str,
    training_id: str,
    artifact_dir: str,
    convergence_data: list[dict] | None = None,
    feature_importances: dict[str, float] | None = None,
    feature_columns: list[str] | None = None,
    root_dir: Path | None = None,
    timestamp: str | None = None,
) -> ModelArtifact:
    """Save a trained model + metadata JSON to resources/trained-models/."""
    if root_dir is None:
        root_dir = _get_default_root()
    root_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{target}_{algorithm}_{ts}.joblib"
    model_path = root_dir / filename
    joblib.dump(model, model_path)

    metadata: dict[str, Any] = {
        "training_id": training_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "source_artifact_dir": artifact_dir,
        "targets": [target],
        "algorithms": {target: algorithm},
        "hyperparameters": {target: hyperparameters},
        "metrics": {target: metrics},
        "feature_version": "v4",
        "feature_dim": feature_dim,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "supported_elements": supported_elements,
        "model_files": [filename],
    }
    if convergence_data:
        metadata["convergence"] = {target: convergence_data}
    if feature_importances:
        metadata["feature_importances"] = {target: feature_importances}
    if feature_columns:
        metadata["feature_columns"] = feature_columns

    metadata_path = root_dir / f"metadata_{target}_{algorithm}_{ts}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return ModelArtifact(
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
