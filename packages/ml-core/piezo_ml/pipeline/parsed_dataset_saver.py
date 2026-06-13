from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ParsedDatasetArtifact:
    artifact_dir: Path
    source_with_uid_path: Path
    parsed_compositions_path: Path
    feature_vectors_path: Path
    preprocessing_log_path: Path


def save_parsed_dataset_artifacts(
    dataset_id: str,
    source_frame: pd.DataFrame,
    parsed_compositions: pd.DataFrame,
    feature_vectors: pd.DataFrame,
    preprocessing_log: list[str],
    root_dir: Path | None = None,
) -> ParsedDatasetArtifact:
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[4] / "resources/training-artifacts"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = root_dir / f"dataset_{dataset_id}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    source_path = artifact_dir / "source_with_uid.csv"
    parsed_path = artifact_dir / "parsed_compositions.csv"
    features_path = artifact_dir / "feature_vectors.csv"
    log_path = artifact_dir / "preprocessing_log.txt"

    source_frame.to_csv(source_path, index=False)
    parsed_compositions.to_csv(parsed_path, index=False)
    feature_vectors.to_csv(features_path, index=False)
    log_path.write_text("\n".join(preprocessing_log) + "\n", encoding="utf-8")

    return ParsedDatasetArtifact(
        artifact_dir=artifact_dir,
        source_with_uid_path=source_path,
        parsed_compositions_path=parsed_path,
        feature_vectors_path=features_path,
        preprocessing_log_path=log_path,
    )
