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
from piezo_ml.models.use_case_mapper import map_use_case, predict_usage
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

        # Map use-case with composite awareness
        use_case = None
        usage_data = None
        if result.status == "success":
            usage = predict_usage(
                d33=result.d33, tc=result.tc, hardness=result.hardness,
                is_composite=result.is_composite,
            )
            if usage.recommendations:
                top = usage.recommendations[0]
                use_case = {
                    "name": top.name, "category": top.category,
                    "confidence": top.confidence, "description": top.description,
                    "icon": top.icon, "color": top.color,
                    "tier": top.tier, "tier_label": top.tier_label,
                    "driving_properties": top.driving_properties,
                    "score": top.score,
                }
            # Include full usage data for richer UI
            usage_data = {
                "recommendations": [
                    {
                        "name": r.name, "score": r.score,
                        "confidence": r.confidence,
                        "tier": r.tier, "tier_label": r.tier_label,
                        "description": r.description,
                        "icon": r.icon, "color": r.color,
                        "driving_properties": r.driving_properties,
                        "category": r.category,
                    }
                    for r in usage.recommendations
                ],
                "caution_notes": usage.caution_notes,
                "property_completeness": usage.property_completeness,
                "properties_used": usage.properties_used,
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
            "hardness": _prop_dict(result.hardness, result.hardness_ci_lower, result.hardness_ci_upper),
            "use_case": use_case,
            "usage_predictions": usage_data,
            "composite_params": result.composite_params,
        }

    # ---- batch prediction from CSV ----

    async def predict_batch_csv(
        self, db: AsyncSession, model_ids: dict[str, str],
        csv_content: bytes, filename: str,
    ) -> dict[str, Any]:
        """Run batch prediction from uploaded CSV with multi-model support."""
        # Load all selected models
        loaded_models = {}
        for target, model_id in model_ids.items():
            if model_id:
                model_record, metadata = await self.get_model_with_metadata(db, model_id)
                loaded_models[target] = (model_record, metadata)

        if not loaded_models:
            raise ValueError("No models selected for batch prediction")

        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(csv_content))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

        # Use first model for batch DB record
        first_record = list(loaded_models.values())[0][0]
        return await self._run_batch_multi(db, loaded_models, first_record, df, filename)

    # ---- batch prediction from existing dataset ----

    async def predict_batch_from_dataset(
        self, db: AsyncSession, model_ids: dict[str, str], dataset_id: str,
    ) -> dict[str, Any]:
        """Run batch prediction using an existing dataset with multi-model support."""
        # Load all selected models
        loaded_models = {}
        for target, model_id in model_ids.items():
            if model_id:
                model_record, metadata = await self.get_model_with_metadata(db, model_id)
                loaded_models[target] = (model_record, metadata)

        if not loaded_models:
            raise ValueError("No models selected for batch prediction")

        # Load dataset materials
        result = await db.execute(
            select(Material)
            .where(Material.dataset_id == dataset_id)
            .order_by(Material.uid)
        )
        materials = result.scalars().all()
        if not materials:
            raise ValueError(f"No materials found in dataset {dataset_id}")

        # Convert to DataFrame — only include formula + composite columns
        rows = []
        for m in materials:
            row = {"uid": m.uid, "formula": m.formula}
            for field in [
                "filler_wt_pct", "matrix_type", "particle_morphology",
                "particle_size_nm", "surface_treatment", "fabrication_method",
            ]:
                row[field] = getattr(m, field, None)
            rows.append(row)
        df = pd.DataFrame(rows)

        # Get dataset display name for filename
        ds = await db.get(Dataset, dataset_id)
        filename = ds.display_name if ds else f"dataset_{dataset_id}"

        first_record = list(loaded_models.values())[0][0]
        return await self._run_batch_multi(db, loaded_models, first_record, df, filename)

    async def _run_batch_multi(
        self, db: AsyncSession,
        loaded_models: dict[str, tuple[TrainedModel, dict[str, Any]]],
        primary_record: TrainedModel,
        df: pd.DataFrame, filename: str,
    ) -> dict[str, Any]:
        """Core batch prediction logic with multi-target support."""
        from piezo_ml.models.use_case_mapper import predict_usage

        # Create batch record
        batch = PredictionBatch(
            model_id=primary_record.id,
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

        # Result rows for CSV + tabular preview
        result_rows = []
        tabular_results = []

        for idx, row in df.iterrows():
            formula_val = str(row.get(formula_col, "")).strip()
            composite_params = _extract_composite_params(row)
            is_composite = _detect_composite_from_row(row)

            if not formula_val or formula_val.lower() in ("nan", "none", ""):
                # Build empty result row
                csv_row = {
                    "uid": row.get("uid", idx + 1),
                    "formula": formula_val,
                    "is_composite": is_composite,
                    "prediction_status": "parse_error",
                    "prediction_notes": "Empty formula",
                }
                result_rows.append(csv_row)
                tabular_results.append({
                    "uid": csv_row["uid"],
                    "formula": formula_val,
                    "is_composite": is_composite,
                    "prediction_status": "parse_error",
                    "prediction_notes": "Empty formula",
                })
                errors += 1
                continue

            # Predict with each selected model
            d33_val = d33_ci_lo = d33_ci_hi = None
            tc_val = tc_ci_lo = tc_ci_hi = None
            hv_val = hv_ci_lo = hv_ci_hi = None
            row_status = "success"
            row_notes_parts = []

            for target, (model_rec, meta) in loaded_models.items():
                try:
                    pred = self.engine.predict_single(
                        formula=formula_val,
                        model_file_path=model_rec.model_file_path,
                        metadata=meta,
                        composite_params=composite_params if is_composite else None,
                    )
                    if pred.status != "success":
                        row_notes_parts.append(f"{target}: {pred.notes or 'failed'}")
                        continue

                    if pred.d33 is not None:
                        d33_val = pred.d33
                        d33_ci_lo = pred.d33_ci_lower
                        d33_ci_hi = pred.d33_ci_upper
                    if pred.tc is not None:
                        tc_val = pred.tc
                        tc_ci_lo = pred.tc_ci_lower
                        tc_ci_hi = pred.tc_ci_upper
                    if pred.hardness is not None:
                        hv_val = pred.hardness
                        hv_ci_lo = pred.hardness_ci_lower
                        hv_ci_hi = pred.hardness_ci_upper
                except Exception as e:
                    row_notes_parts.append(f"{target}: {e}")

            # Determine overall status
            any_predicted = d33_val is not None or tc_val is not None or hv_val is not None
            if any_predicted:
                row_status = "success"
                success += 1
            else:
                row_status = "parse_error"
                errors += 1

            row_notes = "; ".join(row_notes_parts) if row_notes_parts else None

            # Use-case mapping
            top_use_case = None
            use_case_score = None
            if any_predicted:
                usage = predict_usage(d33=d33_val, tc=tc_val, hardness=hv_val, is_composite=is_composite)
                if usage.recommendations:
                    top_use_case = usage.recommendations[0].name
                    use_case_score = usage.recommendations[0].score

            # Build clean CSV row (no source d33/tc/hardness — those are what we predict)
            csv_row = {
                "uid": row.get("uid", idx + 1),
                "formula": formula_val,
                "is_composite": is_composite,
            }
            if d33_val is not None:
                csv_row["d33_predicted"] = d33_val
                csv_row["d33_ci_lower"] = d33_ci_lo
                csv_row["d33_ci_upper"] = d33_ci_hi
            if tc_val is not None:
                csv_row["tc_predicted"] = tc_val
                csv_row["tc_ci_lower"] = tc_ci_lo
                csv_row["tc_ci_upper"] = tc_ci_hi
            if hv_val is not None:
                csv_row["hardness_predicted"] = hv_val
                csv_row["hardness_ci_lower"] = hv_ci_lo
                csv_row["hardness_ci_upper"] = hv_ci_hi
            if top_use_case:
                csv_row["top_use_case"] = top_use_case
                csv_row["use_case_score"] = use_case_score
            csv_row["prediction_status"] = row_status
            csv_row["prediction_notes"] = row_notes or ""
            result_rows.append(csv_row)

            # Tabular result for frontend
            tabular_results.append({
                "uid": csv_row["uid"],
                "formula": formula_val,
                "is_composite": is_composite,
                "d33_predicted": d33_val,
                "d33_ci_lower": d33_ci_lo,
                "d33_ci_upper": d33_ci_hi,
                "tc_predicted": tc_val,
                "tc_ci_lower": tc_ci_lo,
                "tc_ci_upper": tc_ci_hi,
                "hardness_predicted": hv_val,
                "hardness_ci_lower": hv_ci_lo,
                "hardness_ci_upper": hv_ci_hi,
                "top_use_case": top_use_case,
                "use_case_score": use_case_score,
                "prediction_status": row_status,
                "prediction_notes": row_notes,
            })

            # Store in DB (using primary model)
            prediction = Prediction(
                model_id=primary_record.id,
                batch_id=batch.id,
                formula=formula_val,
                is_composite=is_composite,
                composite_params=composite_params if is_composite else None,
                d33_predicted=d33_val,
                d33_ci_lower=d33_ci_lo,
                d33_ci_upper=d33_ci_hi,
                tc_predicted=tc_val,
                tc_ci_lower=tc_ci_lo,
                tc_ci_upper=tc_ci_hi,
                hardness_predicted=hv_val,
                prediction_status=row_status,
                prediction_notes=row_notes,
            )
            db.add(prediction)

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
            "results": tabular_results,
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
