"""
Piezo.AI — Dashboard Service
===============================
Business logic for dashboard: system stats, model management, report generation.
NOTE: This is a DUMB PIPE — no ML logic. Report generation delegates to ml-core.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from piezo_db.models import (
    Dataset,
    Material,
    Prediction,
    PredictionBatch,
    TrainedModel,
    TrainingJob,
)


class DashboardService:
    """Dashboard business logic: stats queries, model CRUD, report orchestration."""

    # ── System Stats ──────────────────────────────────────

    async def get_system_stats(self, db: AsyncSession) -> dict[str, Any]:
        """Gather system-wide counts and DB size."""
        # Dataset counts
        ds_result = await db.execute(
            select(
                func.count(Dataset.id).label("total"),
                func.count(Dataset.id).filter(Dataset.status == "ready").label("ready"),
                func.count(Dataset.id).filter(Dataset.status == "pending").label("pending"),
            )
        )
        ds_row = ds_result.one()

        # Total material rows
        mat_result = await db.execute(select(func.count(Material.id)))
        material_count = mat_result.scalar() or 0

        # Model count
        model_result = await db.execute(select(func.count(TrainedModel.id)))
        model_count = model_result.scalar() or 0

        # Prediction count
        pred_result = await db.execute(select(func.count(Prediction.id)))
        pred_count = pred_result.scalar() or 0

        # Training job counts
        job_result = await db.execute(
            select(
                func.count(TrainingJob.id).label("total"),
                func.count(TrainingJob.id).filter(TrainingJob.status == "completed").label("completed"),
                func.count(TrainingJob.id).filter(TrainingJob.status == "failed").label("failed"),
            )
        )
        job_row = job_result.one()

        # DB size (PostgreSQL specific)
        db_size_mb = 0.0
        try:
            size_result = await db.execute(
                text("SELECT pg_database_size(current_database())")
            )
            db_size_bytes = size_result.scalar() or 0
            db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
        except Exception:
            pass  # Non-critical — skip on error

        return {
            "dataset_count": ds_row.total,
            "dataset_ready_count": ds_row.ready,
            "dataset_pending_count": ds_row.pending,
            "total_material_rows": material_count,
            "trained_model_count": model_count,
            "prediction_count": pred_count,
            "training_job_count": job_row.total,
            "training_completed_count": job_row.completed,
            "training_failed_count": job_row.failed,
            "db_size_mb": db_size_mb,
        }

    # ── Model Management ──────────────────────────────────

    async def list_models(self, db: AsyncSession) -> list[TrainedModel]:
        """List all trained models ordered by creation date."""
        result = await db.execute(
            select(TrainedModel).order_by(TrainedModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_target_distribution(self, db: AsyncSession) -> list[dict]:
        """Get model count per target for donut chart."""
        result = await db.execute(
            select(
                TrainedModel.target,
                func.count(TrainedModel.id).label("count"),
            ).group_by(TrainedModel.target)
        )
        rows = result.all()
        total = sum(r.count for r in rows) or 1
        return [
            {
                "target": r.target,
                "count": r.count,
                "percentage": round(r.count / total * 100, 1),
            }
            for r in rows
        ]

    async def rename_model(
        self, db: AsyncSession, model_id: str, new_name: str,
    ) -> TrainedModel:
        """Rename model display name — UUID stays constant."""
        model = await db.get(TrainedModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        model.display_name = new_name
        return model

    async def set_default_model(
        self, db: AsyncSession, model_id: str,
    ) -> TrainedModel:
        """Set model as default for its target (unsets previous)."""
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

    async def delete_model(
        self, db: AsyncSession, model_id: str,
    ) -> None:
        """Delete a single model and its filesystem artifact.
        
        Must explicitly delete predictions + prediction_batches first
        because the FK constraints don't have ON DELETE CASCADE at DB level.
        """
        model = await db.get(TrainedModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        # Delete related predictions first (FK constraint)
        await db.execute(
            delete(Prediction).where(Prediction.model_id == model_id)
        )
        # Delete related prediction batches (FK constraint)
        await db.execute(
            delete(PredictionBatch).where(PredictionBatch.model_id == model_id)
        )
        await db.flush()
        # Remove model file from disk
        self._remove_model_file(model.model_file_path)
        await db.delete(model)

    async def bulk_delete_models(
        self, db: AsyncSession, model_ids: list[str],
    ) -> dict[str, Any]:
        """Delete multiple models."""
        deleted = 0
        errors: list[str] = []
        for mid in model_ids:
            try:
                await self.delete_model(db, mid)
                deleted += 1
            except Exception as e:
                errors.append(f"{mid}: {e}")
        return {"deleted_count": deleted, "errors": errors}

    def _remove_model_file(self, model_file_path: str) -> None:
        """Remove .joblib model file from filesystem."""
        try:
            path = Path(model_file_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[5] / model_file_path
            if path.exists():
                path.unlink()
        except Exception:
            pass  # Non-critical — log but don't crash

    # ── Dataset Management (for dashboard) ───────────────

    async def list_datasets_summary(self, db: AsyncSession) -> list[dict]:
        """List all datasets with summary info."""
        result = await db.execute(
            select(Dataset).order_by(Dataset.uploaded_at.desc())
        )
        datasets = result.scalars().all()
        summaries = []
        for ds in datasets:
            summaries.append({
                "id": str(ds.id),
                "display_name": ds.display_name,
                "original_filename": ds.original_filename,
                "status": ds.status,
                "total_rows": ds.total_rows,
                "total_columns": ds.total_columns,
                "has_composite_fields": ds.has_composite_fields,
                "uploaded_at": ds.uploaded_at.isoformat() if ds.uploaded_at else "",
                "updated_at": ds.updated_at.isoformat() if ds.updated_at else "",
            })
        return summaries

    async def rename_dataset(
        self, db: AsyncSession, dataset_id: str, new_name: str,
    ) -> dict:
        """Rename a dataset (UUID stays constant)."""
        ds = await db.get(Dataset, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        ds.display_name = new_name
        await db.flush()
        return {
            "id": str(ds.id),
            "display_name": ds.display_name,
            "status": ds.status,
        }

    async def copy_dataset(
        self, db: AsyncSession, dataset_id: str, new_name: str | None = None,
    ) -> dict:
        """Copy a dataset with all its materials."""
        ds = await db.get(Dataset, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")

        copy_name = new_name or f"{ds.display_name} (Copy)"
        new_ds = Dataset(
            display_name=copy_name,
            original_filename=ds.original_filename,
            status=ds.status,
            total_rows=ds.total_rows,
            total_columns=ds.total_columns,
            column_mapping=ds.column_mapping,
            has_composite_fields=ds.has_composite_fields,
        )
        db.add(new_ds)
        await db.flush()

        # Copy materials
        mat_result = await db.execute(
            select(Material).where(Material.dataset_id == dataset_id)
        )
        materials = mat_result.scalars().all()
        for m in materials:
            new_mat = Material(
                dataset_id=new_ds.id,
                uid=m.uid,
                formula=m.formula,
                d33=m.d33,
                tc=m.tc,
                vickers_hardness=m.vickers_hardness,
                qm=m.qm,
                kp=m.kp,
                relative_density_pct=m.relative_density_pct,
                sintering_temp_c=m.sintering_temp_c,
                sintering_method=m.sintering_method,
                ceramic_type=m.ceramic_type,
                fabrication_method=m.fabrication_method,
                matrix_type=m.matrix_type,
                filler_wt_pct=m.filler_wt_pct,
                particle_morphology=m.particle_morphology,
                particle_size_nm=m.particle_size_nm,
                surface_treatment=m.surface_treatment,
                source_doi=m.source_doi,
                source_notes=m.source_notes,
                parse_status=m.parse_status,
                parse_warnings=m.parse_warnings,
                source_row=m.source_row,
                parsed_row=m.parsed_row,
            )
            db.add(new_mat)

        await db.flush()
        return {
            "id": str(new_ds.id),
            "display_name": new_ds.display_name,
            "status": new_ds.status,
        }

    async def delete_dataset(
        self, db: AsyncSession, dataset_id: str,
    ) -> None:
        """Delete a dataset and all its materials (cascade)."""
        ds = await db.get(Dataset, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        # Delete associated training jobs' models' predictions first
        job_result = await db.execute(
            select(TrainingJob).where(TrainingJob.dataset_id == dataset_id)
        )
        jobs = job_result.scalars().all()
        for job in jobs:
            model_result = await db.execute(
                select(TrainedModel).where(TrainedModel.training_job_id == job.id)
            )
            models = model_result.scalars().all()
            for model in models:
                await db.execute(
                    delete(Prediction).where(Prediction.model_id == model.id)
                )
                await db.execute(
                    delete(PredictionBatch).where(PredictionBatch.model_id == model.id)
                )
                self._remove_model_file(model.model_file_path)
        await db.flush()
        # Now delete the dataset (cascades to materials and training_jobs)
        await db.delete(ds)

    # ── Parsed Dataset Download ──────────────────────────

    async def get_parsed_dataset_path(
        self, db: AsyncSession, model_id: str,
    ) -> Path | None:
        """Get path to parsed dataset CSV for a trained model."""
        model = await db.get(TrainedModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        artifact_dir = Path(model.artifact_dir)
        if not artifact_dir.is_absolute():
            artifact_dir = Path(__file__).resolve().parents[5] / model.artifact_dir
        parsed_csv = artifact_dir / "parsed_compositions.csv"
        if parsed_csv.exists():
            return parsed_csv
        # Fallback: feature_vectors.csv
        fv_csv = artifact_dir / "feature_vectors.csv"
        if fv_csv.exists():
            return fv_csv
        return None

    # ── Prediction History ────────────────────────────────

    async def get_prediction_history_grouped(
        self, db: AsyncSession, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent predictions grouped by formula+timestamp.

        When the user predicts d33, tc, hardness for the same formula,
        each target creates a separate Prediction row within ~5 seconds.
        This groups them into a single unified item for the report/AI insight.
        """
        from datetime import timedelta

        result = await db.execute(
            select(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit * 3)  # Fetch extra since we'll merge
        )
        raw = list(result.scalars().all())

        # Group by (formula, is_composite, composite_params) within 5s window
        groups: list[dict[str, Any]] = []
        used: set[str] = set()

        for p in raw:
            pid = str(p.id)
            if pid in used:
                continue

            # Find all predictions for same formula within 5s
            group_ids = [pid]
            merged = {
                "id": pid,  # primary id for the group
                "member_ids": [pid],
                "formula": p.formula,
                "is_composite": p.is_composite,
                "composite_params": p.composite_params,
                "d33_predicted": p.d33_predicted,
                "tc_predicted": p.tc_predicted,
                "hardness_predicted": p.hardness_predicted,
                "prediction_status": p.prediction_status,
                "created_at": p.created_at,
            }
            used.add(pid)

            for other in raw:
                oid = str(other.id)
                if oid in used:
                    continue
                if other.formula != p.formula:
                    continue
                if other.is_composite != p.is_composite:
                    continue
                time_diff = abs((other.created_at - p.created_at).total_seconds())
                if time_diff > 10:  # 10s window for grouping
                    continue

                # Merge values
                used.add(oid)
                merged["member_ids"].append(oid)
                if other.d33_predicted is not None and merged["d33_predicted"] is None:
                    merged["d33_predicted"] = other.d33_predicted
                if other.tc_predicted is not None and merged["tc_predicted"] is None:
                    merged["tc_predicted"] = other.tc_predicted
                if other.hardness_predicted is not None and merged["hardness_predicted"] is None:
                    merged["hardness_predicted"] = other.hardness_predicted

            groups.append(merged)
            if len(groups) >= limit:
                break

        return groups

    async def get_prediction_history(
        self, db: AsyncSession, limit: int = 100,
    ) -> list[Prediction]:
        """Get raw recent predictions (for delete operations)."""
        result = await db.execute(
            select(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_prediction(
        self, db: AsyncSession, prediction_id: str,
    ) -> None:
        """Delete a single prediction record."""
        pred = await db.get(Prediction, prediction_id)
        if not pred:
            raise ValueError(f"Prediction {prediction_id} not found")
        await db.delete(pred)

    async def bulk_delete_predictions(
        self, db: AsyncSession, prediction_ids: list[str],
    ) -> dict[str, Any]:
        """Delete multiple prediction records."""
        deleted = 0
        errors: list[str] = []
        for pid in prediction_ids:
            try:
                await self.delete_prediction(db, pid)
                deleted += 1
            except Exception as e:
                errors.append(f"{pid}: {e}")
        return {"deleted_count": deleted, "errors": errors}

    # ── Report Generation ─────────────────────────────────

    async def generate_report(
        self, db: AsyncSession, options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a PDF report using ml-core report builder."""
        from piezo_ml.reporting.report_builder import PiezoReportBuilder

        # Gather data for the report
        report_data = await self._gather_report_data(db, options)

        # Generate PDF
        builder = PiezoReportBuilder()
        report_id = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"piezo_report_{report_id}_{ts}.pdf"

        output_dir = Path(__file__).resolve().parents[5] / "resources" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        builder.build(report_data, str(output_path))

        return {
            "report_id": report_id,
            "filename": filename,
            "download_url": f"/api/v1/dashboard/reports/{report_id}/download",
            "file_path": str(output_path),
            "generated_at": datetime.now().isoformat(),
        }

    async def _gather_report_data(
        self, db: AsyncSession, options: dict[str, Any],
    ) -> dict[str, Any]:
        """Gather all data needed for the report."""
        data: dict[str, Any] = {"options": options}

        # System stats
        data["stats"] = await self.get_system_stats(db)

        # Selected models
        model_ids = options.get("selected_model_ids", [])
        if model_ids:
            result = await db.execute(
                select(TrainedModel).where(TrainedModel.id.in_(model_ids))
            )
            models = result.scalars().all()
            data["models"] = [
                {
                    "display_name": m.display_name,
                    "target": m.target,
                    "algorithm": m.algorithm,
                    "r2_score": m.r2_score,
                    "rmse": m.rmse,
                    "n_train_samples": m.n_train_samples,
                    "n_test_samples": m.n_test_samples,
                    "training_duration_s": m.training_duration_s,
                    "created_at": m.created_at.isoformat() if m.created_at else "",
                }
                for m in models
            ]
        else:
            # Include all models by default
            all_models = await self.list_models(db)
            data["models"] = [
                {
                    "display_name": m.display_name,
                    "target": m.target,
                    "algorithm": m.algorithm,
                    "r2_score": m.r2_score,
                    "rmse": m.rmse,
                    "n_train_samples": m.n_train_samples,
                    "n_test_samples": m.n_test_samples,
                    "training_duration_s": m.training_duration_s,
                    "created_at": m.created_at.isoformat() if m.created_at else "",
                }
                for m in all_models
            ]

        # Selected predictions for material insight
        pred_ids = options.get("selected_prediction_ids", [])
        if pred_ids:
            result = await db.execute(
                select(Prediction).where(Prediction.id.in_(pred_ids))
            )
            preds = result.scalars().all()
            data["predictions"] = [
                {
                    "formula": p.formula,
                    "d33_predicted": float(p.d33_predicted) if p.d33_predicted is not None else None,
                    "tc_predicted": float(p.tc_predicted) if p.tc_predicted is not None else None,
                    "hardness_predicted": float(p.hardness_predicted) if p.hardness_predicted is not None else None,
                    "is_composite": p.is_composite,
                    "prediction_status": p.prediction_status or "success",
                    "created_at": p.created_at.isoformat() if p.created_at else "",
                }
                for p in preds
            ]

        # LLM configuration for AI insights
        if options.get("include_ai_insight", False):
            from app.core.config import settings
            data["llm_config"] = {
                "provider": settings.LLM_PROVIDER or "google",
                "api_key": settings.effective_llm_api_key,
                "model": settings.LLM_MODEL or "gemini-2.0-flash",
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS,
            }

        return data


# Singleton
dashboard_service = DashboardService()
