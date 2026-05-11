"""
Piezo.AI — Dashboard Router
===============================
REST API endpoints for the Dashboard section.
Delegates all business logic to DashboardService.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.dashboard.schemas import (
    BulkDeleteModelsRequest,
    BulkDeleteModelsResponse,
    DashboardModel,
    PredictionHistoryItem,
    ReportGenerateRequest,
    ReportGenerateResponse,
    SystemStats,
    TargetDistribution,
)
from app.modules.dashboard.service import dashboard_service

router = APIRouter()

# ── In-memory report store (maps report_id → file_path) ────
_report_store: dict[str, str] = {}


# ── System Stats ────────────────────────────────────────────

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(db: AsyncSession = Depends(get_db)):
    """Get system-wide statistics (counts, DB size)."""
    stats = await dashboard_service.get_system_stats(db)
    return stats


# ── Target Distribution ────────────────────────────────────

@router.get("/target-distribution", response_model=list[TargetDistribution])
async def get_target_distribution(db: AsyncSession = Depends(get_db)):
    """Get model count per target for donut chart."""
    return await dashboard_service.get_target_distribution(db)


# ── Model Management ───────────────────────────────────────

@router.get("/models", response_model=list[DashboardModel])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all trained models with extended info."""
    models = await dashboard_service.list_models(db)
    return [
        DashboardModel(
            id=str(m.id),
            display_name=m.display_name,
            target=m.target,
            algorithm=m.algorithm,
            r2_score=m.r2_score,
            rmse=m.rmse,
            feature_dim=m.feature_dim,
            n_train_samples=m.n_train_samples,
            n_test_samples=m.n_test_samples,
            training_duration_s=m.training_duration_s,
            is_default=m.is_default,
            dataset_id=str(m.dataset_id),
            artifact_dir=m.artifact_dir,
            model_file_path=m.model_file_path,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.patch("/models/{model_id}/rename", response_model=DashboardModel)
async def rename_model(
    model_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Rename a trained model (UUID stays constant)."""
    new_name = body.get("display_name", "").strip()
    if not new_name:
        raise HTTPException(400, "display_name is required")
    try:
        model = await dashboard_service.rename_model(db, model_id, new_name)
        return DashboardModel(
            id=str(model.id),
            display_name=model.display_name,
            target=model.target,
            algorithm=model.algorithm,
            r2_score=model.r2_score,
            rmse=model.rmse,
            feature_dim=model.feature_dim,
            n_train_samples=model.n_train_samples,
            n_test_samples=model.n_test_samples,
            training_duration_s=model.training_duration_s,
            is_default=model.is_default,
            dataset_id=str(model.dataset_id),
            artifact_dir=model.artifact_dir,
            model_file_path=model.model_file_path,
            created_at=model.created_at,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.patch("/models/{model_id}/default", response_model=DashboardModel)
async def set_default_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Set model as default for its target."""
    try:
        model = await dashboard_service.set_default_model(db, model_id)
        return DashboardModel(
            id=str(model.id),
            display_name=model.display_name,
            target=model.target,
            algorithm=model.algorithm,
            r2_score=model.r2_score,
            rmse=model.rmse,
            feature_dim=model.feature_dim,
            n_train_samples=model.n_train_samples,
            n_test_samples=model.n_test_samples,
            training_duration_s=model.training_duration_s,
            is_default=model.is_default,
            dataset_id=str(model.dataset_id),
            artifact_dir=model.artifact_dir,
            model_file_path=model.model_file_path,
            created_at=model.created_at,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a single trained model."""
    try:
        await dashboard_service.delete_model(db, model_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/models/bulk-delete", response_model=BulkDeleteModelsResponse)
async def bulk_delete_models(
    body: BulkDeleteModelsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple trained models."""
    result = await dashboard_service.bulk_delete_models(db, body.model_ids)
    return result


# ── Parsed Dataset Download ────────────────────────────────

@router.get("/models/{model_id}/parsed-dataset")
async def download_parsed_dataset(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Download parsed dataset CSV for a trained model."""
    try:
        path = await dashboard_service.get_parsed_dataset_path(db, model_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    if path is None or not path.exists():
        raise HTTPException(404, "Parsed dataset not found for this model")

    return FileResponse(
        path=str(path),
        media_type="text/csv",
        filename=path.name,
    )


# ── Model File Download ───────────────────────────────────

@router.get("/models/{model_id}/model-file")
async def download_model_file(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Download the trained model .joblib file."""
    from piezo_db.models import TrainedModel as TM

    model = await db.get(TM, model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    model_path = Path(model.model_file_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[4] / model.model_file_path

    if not model_path.exists():
        raise HTTPException(404, "Model file not found on disk")

    return FileResponse(
        path=str(model_path),
        media_type="application/octet-stream",
        filename=model_path.name,
    )


# ── Prediction History ────────────────────────────────────

@router.get("/predictions/history", response_model=list[PredictionHistoryItem])
async def get_prediction_history(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """Get recent predictions grouped by formula+timestamp for report generation."""
    groups = await dashboard_service.get_prediction_history_grouped(db, limit)
    return [
        PredictionHistoryItem(
            id=g["id"],
            member_ids=g.get("member_ids", [g["id"]]),
            formula=g["formula"],
            is_composite=g["is_composite"],
            composite_params=g.get("composite_params"),
            d33_predicted=g.get("d33_predicted"),
            tc_predicted=g.get("tc_predicted"),
            hardness_predicted=g.get("hardness_predicted"),
            prediction_status=g.get("prediction_status", "success"),
            created_at=g["created_at"],
        )
        for g in groups
    ]


@router.delete("/predictions/{prediction_id}", status_code=204)
async def delete_prediction(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a single prediction record."""
    try:
        await dashboard_service.delete_prediction(db, prediction_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/predictions/bulk-delete")
async def bulk_delete_predictions(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple prediction records."""
    pred_ids = body.get("prediction_ids", [])
    if not pred_ids:
        raise HTTPException(400, "prediction_ids is required")
    result = await dashboard_service.bulk_delete_predictions(db, pred_ids)
    return result


# ── Report Generation ─────────────────────────────────────

@router.post("/reports/generate", response_model=ReportGenerateResponse)
async def generate_report(
    body: ReportGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a premium PDF report."""
    try:
        result = await dashboard_service.generate_report(db, body.model_dump())
        # Store path for download
        _report_store[result["report_id"]] = result["file_path"]
        return ReportGenerateResponse(
            report_id=result["report_id"],
            filename=result["filename"],
            download_url=result["download_url"],
            generated_at=result["generated_at"],
        )
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


@router.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download a generated PDF report."""
    file_path = _report_store.get(report_id)
    if not file_path:
        # Try scanning reports directory
        reports_dir = Path(__file__).resolve().parents[4] / "resources" / "reports"
        for f in reports_dir.glob(f"*{report_id}*"):
            file_path = str(f)
            break

    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, "Report not found")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=Path(file_path).name,
    )


# ── LLM Config Status ─────────────────────────────────────

@router.get("/llm-status")
async def get_llm_status():
    """Check if LLM API key is configured for AI insights."""
    from app.core.config import settings

    has_key = bool(settings.effective_llm_api_key and settings.effective_llm_api_key.strip())
    provider = settings.LLM_PROVIDER or "none"
    model = settings.LLM_MODEL or ""

    return {
        "configured": has_key,
        "provider": provider if has_key else "none",
        "model": model if has_key else "",
    }


# ── Parsed Compositions for Comparison View ────────────────

@router.get("/datasets/{dataset_id}/parsed-compositions")
async def get_parsed_compositions(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get parsed elemental compositions for a dataset.
    Scans training artifacts for parsed_compositions.csv matching this dataset_id.
    Returns the CSV content as JSON rows for the comparison view.
    """
    import csv
    import io

    # Look for artifact directory matching this dataset
    artifacts_root = Path(__file__).resolve().parents[4] / "resources" / "training-artifacts"
    if not artifacts_root.exists():
        return {"rows": [], "columns": [], "found": False}

    # Search for directory matching dataset_id
    parsed_path = None
    for d in sorted(artifacts_root.iterdir(), reverse=True):
        if d.is_dir() and dataset_id in d.name:
            candidate = d / "parsed_compositions.csv"
            if candidate.exists():
                parsed_path = candidate
                break

    if not parsed_path:
        return {"rows": [], "columns": [], "found": False}

    # Read CSV and return as JSON
    try:
        with open(parsed_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = []
            for row in reader:
                # Convert numeric strings to numbers
                clean = {}
                for k, v in row.items():
                    if v is None or v == "":
                        clean[k] = None
                    else:
                        try:
                            clean[k] = float(v)
                        except ValueError:
                            clean[k] = v
                rows.append(clean)
            return {"rows": rows, "columns": list(columns), "found": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to read parsed compositions: {e}")


# ── Dataset Management (from dashboard) ──────────────────

@router.patch("/datasets/{dataset_id}/rename")
async def rename_dataset(
    dataset_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Rename a dataset from dashboard (UUID stays constant)."""
    new_name = body.get("display_name", "").strip()
    if not new_name:
        raise HTTPException(400, "display_name is required")
    try:
        result = await dashboard_service.rename_dataset(db, dataset_id, new_name)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/datasets/{dataset_id}/copy")
async def copy_dataset(
    dataset_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Copy a dataset with all its materials."""
    new_name = body.get("new_name")
    try:
        result = await dashboard_service.copy_dataset(db, dataset_id, new_name)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a dataset and all its materials."""
    try:
        await dashboard_service.delete_dataset(db, dataset_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── Parse Dataset on Demand ──────────────────────────────

@router.post("/datasets/{dataset_id}/parse")
async def parse_dataset(
    dataset_id: str,
    body: dict | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Parse a dataset's formulas into elemental decomposition on demand.
    This allows viewing parsed data before training.
    Returns the parsed rows with element fractions and carried-over properties.
    """
    import pandas as pd
    from piezo_db.models import Material

    strict_mode = (body or {}).get("strict_mode", True)

    # Load materials from DB
    result = await db.execute(
        select(Material).where(Material.dataset_id == dataset_id).order_by(Material.uid)
    )
    materials = result.scalars().all()
    if not materials:
        raise HTTPException(404, "No materials found for this dataset")

    # Build DataFrame from materials
    rows_data = []
    for m in materials:
        row = {
            "uid": m.uid,
            "formula": m.formula,
            "d33": m.d33,
            "tc": m.tc,
            "vickers_hardness": m.vickers_hardness,
            "qm": m.qm,
            "kp": m.kp,
            "relative_density_pct": m.relative_density_pct,
            "sintering_temp_c": m.sintering_temp_c,
            "sintering_method": m.sintering_method,
            "ceramic_type": m.ceramic_type,
            "fabrication_method": m.fabrication_method,
            "matrix_type": m.matrix_type,
            "filler_wt_pct": m.filler_wt_pct,
            "particle_morphology": m.particle_morphology,
            "particle_size_nm": m.particle_size_nm,
            "surface_treatment": m.surface_treatment,
        }
        rows_data.append(row)

    df = pd.DataFrame(rows_data)

    # Parse formulas
    from piezo_ml.features.feature_engineer import FeatureEngineer
    engineer = FeatureEngineer()
    if strict_mode:
        engineer.parser.strict_mode = True

    try:
        _, parsed_df = engineer.engineer_dataframe(df)
    except Exception as e:
        raise HTTPException(500, f"Parsing failed: {e}")

    # Get skipped rows info
    skipped = getattr(engineer, "_last_skipped_uids", [])

    # Convert to JSON-safe format
    parsed_rows = []
    columns = list(parsed_df.columns) if not parsed_df.empty else []
    for _, row in parsed_df.iterrows():
        clean = {}
        for col in columns:
            val = row[col]
            if pd.isna(val):
                clean[col] = None
            elif isinstance(val, float):
                clean[col] = round(val, 6) if val != 0.0 else 0.0
            else:
                clean[col] = val
        parsed_rows.append(clean)

    return {
        "rows": parsed_rows,
        "columns": columns,
        "total_parsed": len(parsed_rows),
        "total_skipped": len(skipped),
        "skipped_details": [{"uid": uid, "reason": reason} for uid, reason in skipped],
    }


