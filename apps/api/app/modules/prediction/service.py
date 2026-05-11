"""
Prediction Service — orchestrates predictions via ML-Core InferenceEngine.

Loads trained models, caches them, runs single and batch predictions,
and persists results to the database.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from piezo_db.models import (
    Dataset,
    Material,
    Prediction,
    PredictionBatch,
    TrainedModel,
)
from piezo_ml.models.inference_engine import InferenceEngine, PredictionResult
from piezo_ml.models.use_case_mapper import map_use_case
from piezo_ml.parsers import FormulaParser
from piezo_ml.registry import get_supported_elements_list


class PredictionService:
    """Manages predictions: single, batch, and from existing datasets."""

    def __init__(self) -> None:
        self.engine = InferenceEngine()
        self.parser = FormulaParser()

    # ---- model loading helpers ----

    async def get_model_with_metadata(
        self, db: AsyncSession, model_id: str,
    ) -> tuple[TrainedModel, dict[str, Any]]:
        """Load trained model record + metadata JSON from filesystem."""
        model_record = await db.get(TrainedModel, model_id)
        if not model_record:
            raise ValueError(f"Trained model {model_id} not found")

        # Load metadata from filesystem
        metadata = self._load_metadata_for_model(model_record)
        return model_record, metadata

    def _load_metadata_for_model(self, model_record: TrainedModel) -> dict[str, Any]:
        """Load metadata JSON for a trained model."""
        import json

        model_path = Path(model_record.model_file_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[4] / model_record.model_file_path

        # Find corresponding metadata JSON
        models_dir = model_path.parent
        # Try pattern: metadata_<target>_<algo>_<ts>.json
        stem = model_path.stem  # model_d33_xgboost_20260507_143022
        parts = stem.replace("model_", "metadata_")
        metadata_path = models_dir / f"{parts}.json"

        if not metadata_path.exists():
            # Scan for any metadata file matching target + algorithm
            for json_file in models_dir.glob("metadata_*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if model_path.name in data.get("model_files", []):
                        return data
                except Exception:
                    continue
            # Fallback: construct from DB record
            return {
                "targets": [model_record.target],
                "algorithms": {model_record.target: model_record.algorithm},
                "feature_dim": model_record.feature_dim,
                "supported_elements": model_record.supported_elements or [],
                "feature_columns": [],
            }

        return json.loads(metadata_path.read_text(encoding="utf-8"))

    # ---- single prediction ----

    async def predict_single(
        self, db: AsyncSession, model_id: str, formula: str,
        composite_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single prediction and store in DB."""
        model_record, metadata = await self.get_model_with_metadata(db, model_id)

        result = self.engine.predict_single(
            formula=formula,
            model_file_path=model_record.model_file_path,
            metadata=metadata,
            composite_params=composite_params,
        )

        # Map use-case
        use_case = None
        if result.status == "success":
            uc = map_use_case(d33=result.d33, tc=result.tc, hardness=result.hardness)
            use_case = {
                "name": uc.name, "category": uc.category,
                "confidence": uc.confidence, "description": uc.description,
                "icon": uc.icon, "color": uc.color,
            }

        # Persist to DB
        prediction = Prediction(
            model_id=model_record.id,
            formula=formula,
            is_composite=result.is_composite,
            composite_params=composite_params,
            d33_predicted=result.d33,
            d33_ci_lower=result.d33_ci_lower,
            d33_ci_upper=result.d33_ci_upper,
            tc_predicted=result.tc,
            tc_ci_lower=result.tc_ci_lower,
            tc_ci_upper=result.tc_ci_upper,
            hardness_predicted=result.hardness,
            prediction_status=result.status,
            prediction_notes=result.notes,
        )
        db.add(prediction)

        return {
            "formula": result.formula,
            "is_composite": result.is_composite,
            "status": result.status,
            "notes": result.notes,
            "d33": _prop_dict(result.d33, result.d33_ci_lower, result.d33_ci_upper),
            "tc": _prop_dict(result.tc, result.tc_ci_lower, result.tc_ci_upper),
            "hardness": _prop_dict(result.hardness, None, None),
            "use_case": use_case,
            "composite_params": result.composite_params,
        }

    # ---- batch prediction from CSV ----

    async def predict_batch_csv(
        self, db: AsyncSession, model_id: str,
        csv_content: bytes, filename: str,
    ) -> dict[str, Any]:
        """Run batch prediction from uploaded CSV."""
        model_record, metadata = await self.get_model_with_metadata(db, model_id)

        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(csv_content))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

        return await self._run_batch(db, model_record, metadata, df, filename)

    # ---- batch prediction from existing dataset ----

    async def predict_batch_from_dataset(
        self, db: AsyncSession, model_id: str, dataset_id: str,
    ) -> dict[str, Any]:
        """Run batch prediction using an existing dataset."""
        model_record, metadata = await self.get_model_with_metadata(db, model_id)

        # Load dataset materials
        result = await db.execute(
            select(Material)
            .where(Material.dataset_id == dataset_id)
            .order_by(Material.uid)
        )
        materials = result.scalars().all()
        if not materials:
            raise ValueError(f"No materials found in dataset {dataset_id}")

        # Convert to DataFrame
        rows = []
        for m in materials:
            row = {"uid": m.uid, "formula": m.formula}
            for field in [
                "d33", "tc", "vickers_hardness", "filler_wt_pct",
                "matrix_type", "particle_morphology", "particle_size_nm",
                "surface_treatment", "fabrication_method",
            ]:
                row[field] = getattr(m, field, None)
            rows.append(row)
        df = pd.DataFrame(rows)

        # Get dataset display name for filename
        ds = await db.get(Dataset, dataset_id)
        filename = ds.display_name if ds else f"dataset_{dataset_id}"

        return await self._run_batch(db, model_record, metadata, df, filename)

    async def _run_batch(
        self, db: AsyncSession, model_record: TrainedModel,
        metadata: dict[str, Any], df: pd.DataFrame, filename: str,
    ) -> dict[str, Any]:
        """Core batch prediction logic."""
        # Create batch record
        batch = PredictionBatch(
            model_id=model_record.id,
            source_filename=filename,
            total_rows=len(df),
        )
        db.add(batch)
        await db.flush()

        batch_id = str(batch.id)
        success = 0
        errors = 0

        # Detect formula column
        formula_col = _find_formula_column(df)
        if not formula_col:
            raise ValueError("No 'formula' column found in dataset")

        # Result rows for CSV output
        result_rows = []

        for idx, row in df.iterrows():
            formula_val = str(row.get(formula_col, "")).strip()
            composite_params = _extract_composite_params(row)
            is_composite = _detect_composite_from_row(row)

            if not formula_val or formula_val.lower() in ("nan", "none", ""):
                pred = PredictionResult(
                    formula=formula_val, is_composite=is_composite,
                    status="parse_error", notes="Empty formula",
                )
            else:
                pred = self.engine.predict_single(
                    formula=formula_val,
                    model_file_path=model_record.model_file_path,
                    metadata=metadata,
                    composite_params=composite_params if is_composite else None,
                )

            # Store in DB
            prediction = Prediction(
                model_id=model_record.id,
                batch_id=batch.id,
                formula=formula_val,
                is_composite=is_composite,
                composite_params=composite_params if is_composite else None,
                d33_predicted=pred.d33,
                d33_ci_lower=pred.d33_ci_lower,
                d33_ci_upper=pred.d33_ci_upper,
                tc_predicted=pred.tc,
                tc_ci_lower=pred.tc_ci_lower,
                tc_ci_upper=pred.tc_ci_upper,
                hardness_predicted=pred.hardness,
                prediction_status=pred.status,
                prediction_notes=pred.notes,
            )
            db.add(prediction)

            if pred.status == "success":
                success += 1
            else:
                errors += 1

            # Build result row (original data + predictions)
            result_row = row.to_dict()
            target = metadata.get("targets", ["d33"])[0]
            if target == "d33":
                result_row["d33_predicted"] = pred.d33
            elif target == "tc":
                result_row["tc_predicted"] = pred.tc
            elif target == "vickers_hardness":
                result_row["hardness_predicted"] = pred.hardness
            result_row["prediction_status"] = pred.status
            result_row["prediction_notes"] = pred.notes or ""
            result_rows.append(result_row)

        # Save result CSV
        result_df = pd.DataFrame(result_rows)
        result_path = self._save_batch_result(batch_id, result_df)

        batch.success_count = success
        batch.error_count = errors
        batch.result_file_path = str(result_path)

        return {
            "batch_id": batch_id,
            "total_rows": len(df),
            "success_count": success,
            "error_count": errors,
            "result_file_path": str(result_path),
            "source_filename": filename,
        }

    def _save_batch_result(self, batch_id: str, df: pd.DataFrame) -> Path:
        """Save batch prediction results to CSV."""
        output_dir = (
            Path(__file__).resolve().parents[4]
            / "resources" / "prediction-results"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"batch_{batch_id[:8]}_{ts}.csv"
        df.to_csv(path, index=False)
        return path

    # ---- formula validation ----

    def validate_formula(self, formula: str, strict_mode: bool = False) -> dict[str, Any]:
        """Validate a formula for real-time UI feedback."""
        if not formula or not formula.strip():
            return {
                "formula": formula, "is_valid": False,
                "error": "Formula is empty", "warnings": [],
            }

        parsed = self.parser.parse(formula, strict_mode=strict_mode)
        return {
            "formula": formula,
            "is_valid": parsed.is_valid,
            "normalized_formula": parsed.normalized_formula if parsed.is_valid else None,
            "elements": parsed.elements if parsed.is_valid else None,
            "unsupported": parsed.unsupported if parsed.unsupported else None,
            "error": parsed.error,
            "warnings": parsed.warnings,
        }

    # ---- model management ----

    async def list_models(self, db: AsyncSession) -> list[TrainedModel]:
        """List all trained models."""
        result = await db.execute(
            select(TrainedModel).order_by(TrainedModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def rename_model(
        self, db: AsyncSession, model_id: str, new_name: str,
    ) -> TrainedModel:
        """Rename model display name (UUID stays constant)."""
        model = await db.get(TrainedModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        model.display_name = new_name
        return model

    async def set_default_model(
        self, db: AsyncSession, model_id: str,
    ) -> TrainedModel:
        """Set model as default for its target (unsets previous default)."""
        model = await db.get(TrainedModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Unset other defaults for same target
        await db.execute(
            update(TrainedModel)
            .where(TrainedModel.target == model.target, TrainedModel.is_default == True)
            .values(is_default=False)
        )
        model.is_default = True
        return model


# Helpers

def _prop_dict(value, ci_lower, ci_upper) -> dict | None:
    if value is None:
        return None
    return {"value": value, "ci_lower": ci_lower, "ci_upper": ci_upper}


def _find_formula_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().strip() in ("formula", "chemical_formula", "composition"):
            return col
    return None


def _extract_composite_params(row) -> dict[str, Any]:
    """Extract composite parameters from a DataFrame row."""
    params = {}
    for field in [
        "matrix_type", "filler_wt_pct", "particle_morphology",
        "particle_size_nm", "surface_treatment", "fabrication_method",
    ]:
        val = row.get(field)
        if val is not None and str(val).lower() not in ("nan", ""):
            params[field] = val
    return params


def _detect_composite_from_row(row) -> bool:
    """Detect if a row represents a composite material."""
    filler = row.get("filler_wt_pct", 0)
    matrix = row.get("matrix_type", "none")
    try:
        filler_val = float(filler) if filler is not None and str(filler).lower() != "nan" else 0
    except (TypeError, ValueError):
        filler_val = 0
    matrix_str = str(matrix).lower() if matrix is not None else "none"
    return filler_val > 0 and matrix_str not in ("none", "nan", "")


# Singleton
prediction_service = PredictionService()
