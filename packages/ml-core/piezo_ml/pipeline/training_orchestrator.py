"""
Training Orchestrator — complete pipeline from raw DataFrame to trained models.

Coordinates: validation → split → clean → impute → engineer → train → save.
Supports cancellation, progress callbacks, and log streaming.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from piezo_ml.features import FeatureEngineer, get_trainable_targets
from piezo_ml.features.field_registry import CATEGORICAL_FIELDS, TARGET_FIELDS
from piezo_ml.models.model_saver import ModelArtifact, save_trained_model
from piezo_ml.models.trainer import ModelTrainer, TrainingCancelledError, TrainingResult
from piezo_ml.models.optuna_tuner import OptunaTuner
from piezo_ml.pipeline.data_cleaner import clean_dataframe
from piezo_ml.pipeline.data_loader import DataLoaderValidator
from piezo_ml.pipeline.missing_value_strategies import apply_missing_value_strategies
from piezo_ml.pipeline.parsed_dataset_saver import save_parsed_dataset_artifacts
from piezo_ml.pipeline.post_parse_validator import validate_post_parse_frame
from piezo_ml.pipeline.sentinel_handler import detect_sentinel_issues
from piezo_ml.registry import get_supported_elements_list


@dataclass
class TrainingConfig:
    """Configuration for one training run."""
    dataset_id: str
    training_id: str
    targets: list[str]
    algorithms: dict[str, str]       # target → algorithm
    hyperparameters: dict[str, dict]  # target → {param: value}
    selected_fields: list[str]
    missing_strategies: dict[str, str]  # field → strategy
    mode: str = "manual"             # "manual" | "auto"


@dataclass
class TrainingOutput:
    """Result of a complete training run."""
    training_id: str
    results: list[TrainingResult]
    model_artifacts: list[ModelArtifact]
    artifact_dir: str
    initial_rows: int
    initial_columns: int
    final_rows: int
    final_columns: int
    logs: list[str] = field(default_factory=list)
    sentinel_issues: list[dict] = field(default_factory=list)


# Stage weights for progress bar (must sum to 100)
STAGE_WEIGHTS = {
    "validating": 5,
    "splitting": 5,
    "cleaning": 10,
    "imputing": 10,
    "engineering": 20,
    "training": 50,
}


class TrainingOrchestrator:
    """Run the full training pipeline with progress and cancellation support."""

    def __init__(
        self,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        convergence_callback: Callable[[str, int, float], None] | None = None,
    ) -> None:
        self.cancel_event = cancel_event
        self.progress_cb = progress_callback
        self.log_cb = log_callback
        self.convergence_cb = convergence_callback
        self._loader = DataLoaderValidator()
        self._engineer = FeatureEngineer()

    def run(self, df: pd.DataFrame, config: TrainingConfig) -> TrainingOutput:
        """Execute the complete training pipeline."""
        logs: list[str] = []
        progress = 0.0

        def _log(level: str, msg: str):
            logs.append(f"[{level.upper()}] {msg}")
            if self.log_cb:
                self.log_cb(level, msg)

        def _progress(pct: float, stage: str):
            nonlocal progress
            progress = pct
            if self.progress_cb:
                self.progress_cb(pct, stage)

        self._check_cancelled()
        initial_rows, initial_cols = df.shape
        _log("info", f"Starting with {initial_rows} rows × {initial_cols} columns")

        # Ensure uid column exists (DataLoaderValidator adds it from CSV,
        # but direct DataFrame input may not have it)
        if "uid" not in df.columns:
            df = df.copy()
            df.insert(0, "uid", range(1, len(df) + 1))

        # ------ 1. Pre-training validation ------
        _progress(0, "Validating dataset")
        _log("info", "Running pre-training validation...")
        sentinel_issues = detect_sentinel_issues(df, config.selected_fields)
        if sentinel_issues:
            for issue in sentinel_issues:
                _log("warning", issue.message)
        _progress(STAGE_WEIGHTS["validating"], "Validation complete")

        # ------ 2. Split train/test ------
        self._check_cancelled()
        _progress(STAGE_WEIGHTS["validating"], "Splitting train/test")
        _log("info", "Splitting train/test (80/20)...")
        train_df, test_df = self._loader.split_train_test(df)
        _log("info", f"Train/test split: train={len(train_df)}, test={len(test_df)}")
        _progress(
            STAGE_WEIGHTS["validating"] + STAGE_WEIGHTS["splitting"],
            "Split complete"
        )

        # ------ 3. Clean ------
        self._check_cancelled()
        _progress(
            STAGE_WEIGHTS["validating"] + STAGE_WEIGHTS["splitting"],
            "Cleaning data"
        )
        _log("info", "Cleaning data (normalizing sentinels, fixing types, dedup)...")
        train_df, train_report = clean_dataframe(train_df)
        test_df, test_report = clean_dataframe(test_df)
        if train_report.dropped_duplicates > 0:
            _log("info", f"Train: dropped {train_report.dropped_duplicates} duplicate rows")
        if test_report.dropped_duplicates > 0:
            _log("info", f"Test: dropped {test_report.dropped_duplicates} duplicate rows")
        _progress(
            sum(STAGE_WEIGHTS[s] for s in ("validating", "splitting", "cleaning")),
            "Cleaning complete"
        )

        # ------ 4. Apply missing-value strategies ------
        self._check_cancelled()
        cum = sum(STAGE_WEIGHTS[s] for s in ("validating", "splitting", "cleaning"))
        _progress(cum, "Applying imputation strategies")
        _log("info", "Applying missing-value strategies...")

        strategies = config.missing_strategies or {}
        if strategies:
            # Only apply strategies for fields that are actually selected
            # (prevents non-selected targets like vickers_hardness from
            # dropping all rows via "drop" strategy)
            relevant_fields = set(config.selected_fields) | set(config.targets)
            filtered_strategies = {
                k: v for k, v in strategies.items() if k in relevant_fields
            }
            if filtered_strategies:
                train_df, mv_report = apply_missing_value_strategies(train_df, filtered_strategies)
                strat_str = ", ".join(f"{k}={v}" for k, v in mv_report.applied_strategies.items())
                _log("info", f"Train strategies: {strat_str}")
                if mv_report.dropped_rows:
                    _log("info", f"Train: dropped {mv_report.dropped_rows} rows via drop strategy")

        # Test set policy: drop rows with missing required values (no imputation)
        required_cols = ["formula"] + config.targets
        test_df, test_log = self._loader.apply_test_set_policy(test_df, required_cols)
        _log("info", test_log)

        cum += STAGE_WEIGHTS["imputing"]
        _progress(cum, "Imputation complete")

        # ------ 5. Feature engineering ------
        self._check_cancelled()
        _progress(cum, "Engineering features")
        _log("info", "Parsing formulas and engineering features...")

        feature_vectors, parsed_compositions = self._engineer.engineer_dataframe(train_df)
        # Log skipped rows from feature engineering
        train_skipped = getattr(self._engineer, '_last_skipped_uids', [])
        if train_skipped:
            _log("warning", f"Train: skipped {len(train_skipped)} rows with parse errors:")
            for uid, reason in train_skipped:
                _log("warning", f"  uid={uid}: {reason}")

        test_features, _ = self._engineer.engineer_dataframe(test_df)
        test_skipped = getattr(self._engineer, '_last_skipped_uids', [])
        if test_skipped:
            _log("warning", f"Test: skipped {len(test_skipped)} rows with parse errors")

        # Align columns: test may have different frac_* columns than train
        # Reindex test to match train columns, filling missing with 0.0
        test_features = test_features.reindex(columns=feature_vectors.columns, fill_value=0.0)

        # ------ 5b. Append composite features ------
        # Encode composite columns from original data so models learn
        # the difference between bulk and composite materials.
        from piezo_ml.features.composite_encoder import (
            encode_composite_row,
            COMPOSITE_FEATURE_COLUMNS,
        )

        _log("info", "Appending composite features to training vectors...")
        _append_composite_features(feature_vectors, train_df, COMPOSITE_FEATURE_COLUMNS, encode_composite_row)
        _append_composite_features(test_features, test_df, COMPOSITE_FEATURE_COLUMNS, encode_composite_row)
        _log("info", f"Composite features appended: {COMPOSITE_FEATURE_COLUMNS}")

        post_issues = validate_post_parse_frame(feature_vectors)
        if post_issues:
            for issue in post_issues:
                _log("warning", f"Post-parse: {issue}")

        # Save artifacts (source + parsed + features)
        cleaned_full, _ = clean_dataframe(df)
        all_features, all_parsed = self._engineer.engineer_dataframe(cleaned_full)
        # Also append composite features to full artifact
        _append_composite_features(all_features, cleaned_full, COMPOSITE_FEATURE_COLUMNS, encode_composite_row)
        artifact = save_parsed_dataset_artifacts(
            dataset_id=config.dataset_id,
            source_frame=cleaned_full,
            parsed_compositions=all_parsed,
            feature_vectors=all_features,
            preprocessing_log=logs.copy(),
        )
        _log("info", f"Artifacts saved: {artifact.artifact_dir}")

        cum += STAGE_WEIGHTS["engineering"]
        _progress(cum, "Feature engineering complete")

        # Align train_df/test_df to only rows that survived parsing
        # (some rows may have been skipped due to invalid formulas)
        if "uid" in feature_vectors.columns:
            train_uids = set(feature_vectors["uid"].values)
            train_df = train_df[train_df["uid"].isin(train_uids)].reset_index(drop=True)
        if "uid" in test_features.columns:
            test_uids = set(test_features["uid"].values)
            test_df = test_df[test_df["uid"].isin(test_uids)].reset_index(drop=True)

        final_rows = len(feature_vectors)
        final_cols = feature_vectors.shape[1] if not feature_vectors.empty else 0
        _log("info", f"Final dataset: {final_rows} train rows × {final_cols} features")

        # ------ 6. Train models ------
        self._check_cancelled()
        _progress(cum, "Training models")

        # Prepare feature columns (exclude uid, formula)
        feature_cols = [c for c in feature_vectors.columns if c not in ("uid", "formula")]
        train_weight = STAGE_WEIGHTS["training"] / max(len(config.targets), 1)

        results: list[TrainingResult] = []
        model_artifacts: list[ModelArtifact] = []
        supported_elements = get_supported_elements_list()

        for i, target in enumerate(config.targets):
            self._check_cancelled()
            algorithm = config.algorithms.get(target, "random_forest")
            hyper = config.hyperparameters.get(target, {})

            _log("info", f"--- Training {algorithm} for {target} ---")
            stage_label = f"Training {algorithm} for {target}"
            _progress(cum + (i * train_weight), stage_label)

            # Extract X, y for this target
            if target not in train_df.columns or target not in test_df.columns:
                _log("error", f"Target '{target}' not found in dataset — skipping")
                continue

            X_train = feature_vectors[feature_cols].values
            y_train = pd.to_numeric(train_df[target], errors="coerce").values

            X_test = test_features[feature_cols].values
            y_test = pd.to_numeric(test_df[target], errors="coerce").values

            # Drop NaN targets
            train_mask = ~np.isnan(y_train)
            test_mask = ~np.isnan(y_test)
            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                _log("error", f"Insufficient data for {target} after filtering — skipping")
                continue

            # Auto-tune if mode is "auto"
            if config.mode == "auto":
                _log("info", f"Running Optuna auto-tune for {algorithm}/{target}...")
                tuner = OptunaTuner(
                    algorithm=algorithm, target=target,
                    cancel_event=self.cancel_event, log_callback=_log,
                )
                hyper = tuner.tune(X_train, y_train)

            # Train
            trainer = ModelTrainer(
                algorithm=algorithm, target=target, hyperparameters=hyper,
                cancel_event=self.cancel_event, log_callback=_log,
            )
            result = trainer.train(X_train, y_train, X_test, y_test, feature_cols)
            result.n_train = len(X_train)
            result.n_test = len(X_test)
            results.append(result)

            # Send convergence data
            if self.convergence_cb and result.convergence_data:
                for point in result.convergence_data:
                    self.convergence_cb(target, point["iteration"], point["metric"])

            # Save model artifact
            ma = save_trained_model(
                model=result.model,
                target=target, algorithm=algorithm,
                metrics={"r2": result.r2, "rmse": result.rmse},
                hyperparameters=result.hyperparameters,
                feature_dim=len(feature_cols),
                n_train=result.n_train, n_test=result.n_test,
                supported_elements=supported_elements,
                dataset_id=config.dataset_id,
                training_id=config.training_id,
                artifact_dir=str(artifact.artifact_dir),
                convergence_data=result.convergence_data,
                feature_importances=result.feature_importances,
                feature_columns=feature_cols,
            )
            model_artifacts.append(ma)
            _log("success", f"Model saved: {ma.model_path.name}")

        cum += STAGE_WEIGHTS["training"]
        _progress(100.0, "Training complete")
        _log("success", "All training complete!")

        sentinel_dicts = [
            {"field": si.field, "issue_type": si.issue_type,
             "count": si.count, "message": si.message}
            for si in sentinel_issues
        ]

        return TrainingOutput(
            training_id=config.training_id,
            results=results,
            model_artifacts=model_artifacts,
            artifact_dir=str(artifact.artifact_dir),
            initial_rows=initial_rows,
            initial_columns=initial_cols,
            final_rows=final_rows,
            final_columns=final_cols,
            logs=logs,
            sentinel_issues=sentinel_dicts,
        )

    def _check_cancelled(self) -> None:
        if self.cancel_event and self.cancel_event.is_set():
            raise TrainingCancelledError("Training cancelled by user")


def _append_composite_features(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    composite_columns: list[str],
    encode_fn,
) -> None:
    """
    Append composite feature columns to the feature DataFrame in-place.

    Aligns by uid: for each row in feature_df, finds matching row in
    source_df and encodes its composite columns.
    """
    from piezo_ml.features.composite_encoder import COMPOSITE_FEATURE_COLUMNS

    # Build encoded composite values for each row in source_df, indexed by uid
    uid_to_composite: dict[int, dict[str, float]] = {}
    for _, row in source_df.iterrows():
        uid = int(row.get("uid", 0))
        uid_to_composite[uid] = encode_fn(row)

    # Initialize composite columns with zeros
    for col in COMPOSITE_FEATURE_COLUMNS:
        feature_df[col] = 0.0

    # Fill in values by matching uid
    if "uid" in feature_df.columns:
        for idx, row in feature_df.iterrows():
            uid = int(row["uid"])
            if uid in uid_to_composite:
                for col, val in uid_to_composite[uid].items():
                    feature_df.at[idx, col] = val
    else:
        # Fallback: assume same row order
        for i, (_, src_row) in enumerate(source_df.iterrows()):
            if i >= len(feature_df):
                break
            encoded = encode_fn(src_row)
            for col, val in encoded.items():
                feature_df.iat[i, feature_df.columns.get_loc(col)] = val

