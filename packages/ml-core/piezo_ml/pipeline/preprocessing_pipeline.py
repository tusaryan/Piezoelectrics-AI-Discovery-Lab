from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from piezo_ml.features import FeatureEngineer
from piezo_ml.pipeline.data_loader import DataLoaderValidator
from piezo_ml.pipeline.missing_value_strategies import apply_missing_value_strategies
from piezo_ml.pipeline.parsed_dataset_saver import ParsedDatasetArtifact, save_parsed_dataset_artifacts
from piezo_ml.pipeline.post_parse_validator import validate_post_parse_frame


@dataclass
class PreprocessingOutput:
    cleaned: pd.DataFrame
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_vectors: pd.DataFrame
    parsed_compositions: pd.DataFrame
    artifact: ParsedDatasetArtifact
    logs: list[str]


class PreprocessingPipeline:
    def __init__(self) -> None:
        self.loader = DataLoaderValidator()
        self.engineer = FeatureEngineer()

    def run(
        self,
        dataset_id: str,
        csv_path: str | Path,
        required_targets: list[str],
        train_missing_strategies: dict[str, str] | None = None,
    ) -> PreprocessingOutput:
        logs: list[str] = []
        raw = self.loader.load_csv(csv_path)
        logs.append(f"Starting with {raw.shape[0]} rows × {raw.shape[1]} columns")

        # Plan-compliant order: split first, then preprocess each split.
        train_frame, test_frame = self.loader.split_train_test(raw)
        logs.append(f"Train/test split: train={len(train_frame)}, test={len(test_frame)}")

        train_frame = self.loader.clean(train_frame)
        test_frame = self.loader.clean(test_frame)
        logs.append(f"After cleaning: train={train_frame.shape[0]}, test={test_frame.shape[0]}")

        cleaned = self.loader.clean(raw)
        logs.append(f"Reference cleaned dataset: {cleaned.shape[0]} rows × {cleaned.shape[1]} columns")

        strategies = train_missing_strategies or {target: "median" for target in required_targets}
        train_frame, strategy_report = apply_missing_value_strategies(train_frame, strategies)
        logs.append(
            "Train missing-value strategies applied: "
            + ", ".join(f"{k}={v}" for k, v in strategy_report.applied_strategies.items())
        )
        if strategy_report.dropped_rows:
            logs.append(f"Train set: dropped {strategy_report.dropped_rows} rows via drop strategy")

        test_frame, test_log = self.loader.apply_test_set_policy(test_frame, ["formula", *required_targets])
        logs.append(test_log)

        feature_vectors, parsed_compositions = self.engineer.engineer_dataframe(cleaned)
        post_parse_issues = validate_post_parse_frame(feature_vectors)
        if post_parse_issues:
            raise ValueError("Post-parse validation failed: " + "; ".join(post_parse_issues))
        logs.append("Post-parse validation passed")

        artifact = save_parsed_dataset_artifacts(
            dataset_id=dataset_id,
            source_frame=cleaned,
            parsed_compositions=parsed_compositions,
            feature_vectors=feature_vectors,
            preprocessing_log=logs,
        )
        logs.append(f"Artifacts saved: {artifact.artifact_dir}")

        return PreprocessingOutput(
            cleaned=cleaned,
            train_frame=train_frame,
            test_frame=test_frame,
            feature_vectors=feature_vectors,
            parsed_compositions=parsed_compositions,
            artifact=artifact,
            logs=logs,
        )
