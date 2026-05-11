"""
Prediction Router — API endpoints for predict, batch, formula validation, model management.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.prediction.schemas import (
    BatchPredictRequest,
    BatchPredictSummary,
    BatchResultRow,
    FormulaValidateRequest,
    FormulaValidateResponse,
    ModelRenameRequest,
    ModelSetDefaultRequest,
    PredictRequest,
    PredictResponse,
    PropertyPrediction,
    SupportedElementsResponse,
    TrainedModelListItem,
    UseCaseInfo,
    UsagePredictionsInfo,
)
from app.modules.prediction.service import prediction_service

from piezo_ml.registry import get_supported_elements_list, get_element_count

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /models — list all trained models
# ---------------------------------------------------------------------------
@router.get("/models", response_model=list[TrainedModelListItem])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all trained models for model selector."""
    models = await prediction_service.list_models(db)
    return [
        TrainedModelListItem(
            id=str(m.id),
            display_name=m.display_name,
            target=m.target,
            algorithm=m.algorithm,
            r2_score=m.r2_score,
            rmse=m.rmse,
            feature_dim=m.feature_dim,
            n_train_samples=m.n_train_samples,
            n_test_samples=m.n_test_samples,
            supported_elements=m.supported_elements or [],
            is_default=m.is_default,
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
        for m in models
    ]


# ---------------------------------------------------------------------------
# PATCH /models/{model_id}/rename — rename model display name
# ---------------------------------------------------------------------------
@router.patch("/models/{model_id}/rename", response_model=TrainedModelListItem)
async def rename_model(
    model_id: UUID,
    body: ModelRenameRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rename a trained model (UUID stays constant)."""
    try:
        m = await prediction_service.rename_model(db, str(model_id), body.display_name)
        return TrainedModelListItem(
            id=str(m.id),
            display_name=m.display_name,
            target=m.target,
            algorithm=m.algorithm,
            r2_score=m.r2_score,
            rmse=m.rmse,
            feature_dim=m.feature_dim,
            n_train_samples=m.n_train_samples,
            n_test_samples=m.n_test_samples,
            supported_elements=m.supported_elements or [],
            is_default=m.is_default,
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# PATCH /models/{model_id}/default — set as default
# ---------------------------------------------------------------------------
@router.patch("/models/{model_id}/default", response_model=TrainedModelListItem)
async def set_default_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Set model as default for its target (unsets previous default)."""
    try:
        m = await prediction_service.set_default_model(db, str(model_id))
        return TrainedModelListItem(
            id=str(m.id),
            display_name=m.display_name,
            target=m.target,
            algorithm=m.algorithm,
            r2_score=m.r2_score,
            rmse=m.rmse,
            feature_dim=m.feature_dim,
            n_train_samples=m.n_train_samples,
            n_test_samples=m.n_test_samples,
            supported_elements=m.supported_elements or [],
            is_default=m.is_default,
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# POST /predict — single prediction
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictResponse)
async def predict_single(
    body: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run a single material property prediction."""
    try:
        result = await prediction_service.predict_single(
            db, str(body.model_id), body.formula, body.composite_params,
        )
        return _format_predict_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# POST /predict/batch — batch prediction from CSV upload
# ---------------------------------------------------------------------------
@router.post("/predict/batch", response_model=BatchPredictSummary)
async def predict_batch(
    model_ids: str = Form(..., description="JSON dict of per-target model IDs"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload CSV and predict all rows with multi-model support."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    import json as json_mod
    try:
        parsed_ids = json_mod.loads(model_ids)
    except Exception:
        raise HTTPException(status_code=400, detail="model_ids must be a valid JSON dict")

    content = await file.read()
    try:
        result = await prediction_service.predict_batch_csv(
            db, parsed_ids, content, file.filename,
        )
        return BatchPredictSummary(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /predict/batch-from-dataset — batch from existing dataset
# ---------------------------------------------------------------------------
@router.post("/predict/batch-from-dataset", response_model=BatchPredictSummary)
async def predict_batch_from_dataset(
    body: BatchPredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run batch prediction using an existing dataset."""
    try:
        result = await prediction_service.predict_batch_from_dataset(
            db, body.model_ids, str(body.dataset_id),
        )
        return BatchPredictSummary(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /validate-formula — real-time formula validation
# ---------------------------------------------------------------------------
@router.post("/validate-formula", response_model=FormulaValidateResponse)
async def validate_formula(body: FormulaValidateRequest):
    """Validate formula for real-time green/red indicator."""
    result = prediction_service.validate_formula(body.formula, strict_mode=body.strict_mode)
    return FormulaValidateResponse(**result)


# ---------------------------------------------------------------------------
# GET /supported-elements — list all supported elements
# ---------------------------------------------------------------------------
@router.get("/supported-elements", response_model=SupportedElementsResponse)
async def get_supported_elements():
    """Return list of all supported elements."""
    elements = get_supported_elements_list()
    return SupportedElementsResponse(elements=elements, count=len(elements))


# ---------------------------------------------------------------------------
# GET /batch/{batch_id}/download — download batch result CSV
# ---------------------------------------------------------------------------
@router.get("/batch/{batch_id}/download")
async def download_batch_result(batch_id: UUID, db: AsyncSession = Depends(get_db)):
    """Download the result CSV of a batch prediction."""
    from fastapi.responses import FileResponse
    from piezo_db.models import PredictionBatch

    batch = await db.get(PredictionBatch, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if not batch.result_file_path:
        raise HTTPException(status_code=404, detail="Result file not available")

    from pathlib import Path
    file_path = Path(batch.result_file_path)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parents[4] / batch.result_file_path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    dl_filename = f"predictions_{batch.source_filename}"
    if not dl_filename.endswith(".csv"):
        dl_filename += ".csv"

    return FileResponse(
        path=str(file_path),
        media_type="text/csv",
        filename=dl_filename,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _format_predict_response(result: dict) -> PredictResponse:
    """Format prediction result dict into PredictResponse."""
    d33_data = result.get("d33")
    tc_data = result.get("tc")
    hardness_data = result.get("hardness")
    uc_data = result.get("use_case")
    usage_data = result.get("usage_predictions")

    # Build usage predictions info
    usage_info = None
    if usage_data:
        recs = []
        for r in usage_data.get("recommendations", []):
            recs.append(UseCaseInfo(
                name=r.get("name", ""),
                category=r.get("category", ""),
                confidence=r.get("confidence", 0),
                description=r.get("description", ""),
                icon=r.get("icon", "🔧"),
                color=r.get("color", "#6366F1"),
                tier=r.get("tier"),
                tier_label=r.get("tier_label"),
                driving_properties=r.get("driving_properties"),
                score=r.get("score"),
            ))
        usage_info = UsagePredictionsInfo(
            recommendations=recs,
            caution_notes=usage_data.get("caution_notes", []),
            property_completeness=usage_data.get("property_completeness", "full"),
            properties_used=usage_data.get("properties_used", []),
        )

    return PredictResponse(
        formula=result["formula"],
        is_composite=result["is_composite"],
        status=result["status"],
        notes=result.get("notes"),
        d33=PropertyPrediction(**d33_data) if d33_data else None,
        tc=PropertyPrediction(**tc_data) if tc_data else None,
        hardness=PropertyPrediction(**hardness_data) if hardness_data else None,
        use_case=UseCaseInfo(**uc_data) if uc_data else None,
        usage_predictions=usage_info,
        composite_params=result.get("composite_params"),
    )
