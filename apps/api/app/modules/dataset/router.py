"""
Piezo.AI — Dataset API Router
================================
13 endpoints for dataset upload, column mapping, material CRUD,
and data quality reporting.

ARCHITECTURAL RULE: DUMB PIPE — all ML logic delegated to ml-core.

Endpoints:
  POST   /upload                          Upload CSV → preview
  POST   /{dataset_id}/map               Apply column mapping → save to DB
  GET    /{dataset_id}/quality-report     Data quality + formula validation
  POST   /{dataset_id}/finalize           Mark dataset as 'ready'
  GET    /                                List all datasets
  GET    /fields                          Get backend field metadata (for mapping UI)
  GET    /{dataset_id}                    Get dataset detail
  PATCH  /{dataset_id}                    Rename dataset
  DELETE /{dataset_id}                    Delete dataset + cascade
  GET    /{dataset_id}/materials          Get materials (paginated, search, sort)
  POST   /{dataset_id}/materials          Add new material row
  PATCH  /{dataset_id}/materials/{mid}    Update single material
  POST   /{dataset_id}/materials/bulk     Bulk update/delete
  DELETE /{dataset_id}/materials/{mid}    Delete single material
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.dataset import service
from app.modules.dataset.schemas import (
    BACKEND_FIELD_META,
    BackendFieldInfo,
    BulkDeleteDatasetsRequest,
    BulkMaterialRequest,
    ColumnMappingRequest,
    DataQualityReport,
    DatasetCopyRequest,
    DatasetDetailResponse,
    DatasetRenameRequest,
    DatasetSummaryResponse,
    MaterialCreateRequest,
    MaterialResponse,
    MaterialUpdateRequest,
    PaginatedMaterialsResponse,
    UploadPreviewResponse,
    BulkUpdateResponse,
)

logger = logging.getLogger("piezo.dataset")

router = APIRouter()


# ---------------------------------------------------------------------------
# Upload Flow
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadPreviewResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV file and get a preview with columns, row count,
    sample rows, and auto-suggested column mappings.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted. Please upload a .csv file.",
        )

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = await service.upload_csv(content, file.filename, db)
        logger.info("[UPLOAD] %s → %d rows, %d columns, dataset_id=%s",
                    file.filename, result.row_count, len(result.columns), result.dataset_id)
        return result
    except ValueError as e:
        logger.error("[UPLOAD] FAILED %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/map", response_model=DatasetDetailResponse)
async def apply_column_mapping(
    dataset_id: str,
    body: ColumnMappingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Apply column mapping to a dataset. Saves all rows to the materials
    table with backend field names, auto-assigns uid (sequential from 1),
    and runs formula validation.
    """
    try:
        result = await service.apply_column_mapping(dataset_id, body.mapping, db)
        logger.info("[MAP] dataset=%s → %d fields mapped, %d rows, composite=%s",
                    dataset_id[:8], len(body.mapping), result.total_rows, result.has_composite_fields)
        return result
    except ValueError as e:
        logger.error("[MAP] FAILED dataset=%s: %s", dataset_id[:8], e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/quality-report", response_model=DataQualityReport)
async def get_quality_report(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate data quality report with issue detection:
    - Missing values
    - Type mismatches
    - Unsupported elements
    - Parse errors
    """
    try:
        result = await service.get_quality_report(dataset_id, db)
        logger.info("[QUALITY] dataset=%s → %d rows, %d valid, %d issues",
                    dataset_id[:8], result.total_rows, result.valid_rows, result.issue_count)
        if result.issue_count > 0:
            for issue in result.issues:
                logger.warning("  ⚠ uid=%d col=%s type=%s: %s",
                               issue.row_uid, issue.column, issue.issue_type, issue.message[:80])
        return result
    except ValueError as e:
        logger.error("[QUALITY] FAILED dataset=%s: %s", dataset_id[:8], e)
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/finalize", response_model=DatasetDetailResponse)
async def finalize_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Mark dataset status as 'ready' after user resolves issues."""
    try:
        result = await service.finalize_dataset(dataset_id, db)
        logger.info("[FINALIZE] dataset=%s → status=%s", dataset_id[:8], result.status)
        return result
    except ValueError as e:
        logger.error("[FINALIZE] FAILED dataset=%s: %s", dataset_id[:8], e)
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[DatasetSummaryResponse])
async def list_datasets(
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded datasets with summary metadata."""
    return await service.list_datasets(db)


@router.get("/fields", response_model=list[BackendFieldInfo])
async def get_backend_fields():
    """
    Get backend field metadata for the column mapping UI.
    Returns categorized field definitions with labels, types, and options.
    """
    return BACKEND_FIELD_META


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full dataset detail including column mapping."""
    try:
        return await service.get_dataset(dataset_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{dataset_id}", response_model=DatasetDetailResponse)
async def rename_dataset(
    dataset_id: str,
    body: DatasetRenameRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rename dataset (display_name only — UUID never changes)."""
    try:
        return await service.rename_dataset(dataset_id, body.display_name, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete dataset and all associated materials (cascade)."""
    try:
        await service.delete_dataset(dataset_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Copy + Bulk Delete
# ---------------------------------------------------------------------------

@router.post("/{dataset_id}/copy", response_model=DatasetDetailResponse)
async def copy_dataset(
    dataset_id: str,
    body: DatasetCopyRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Copy a dataset and all its materials."""
    try:
        new_name = body.new_name if body else None
        result = await service.copy_dataset(dataset_id, new_name, db)
        logger.info("[COPY] dataset=%s → new=%s", dataset_id[:8], result.id[:8])
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/bulk-delete", response_model=BulkUpdateResponse)
async def bulk_delete_datasets(
    body: BulkDeleteDatasetsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple datasets at once."""
    result = await service.bulk_delete_datasets(body.dataset_ids, db)
    return BulkUpdateResponse(
        updated_count=0,
        deleted_count=result["deleted_count"],
        errors=result.get("errors", []),
    )


# ---------------------------------------------------------------------------
# Material CRUD
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/materials", response_model=PaginatedMaterialsResponse)
async def get_materials(
    dataset_id: str,
    search: str | None = Query(None, description="Search by formula (case-insensitive)"),
    sort_by: str = Query("uid", description="Column to sort by"),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=5000, description="Rows per page (max 5000)"),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated materials for a dataset with search and sort."""
    return await service.get_materials(
        dataset_id, db,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size,
    )


@router.post("/{dataset_id}/materials", response_model=MaterialResponse, status_code=201)
async def add_material(
    dataset_id: str,
    body: MaterialCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Add a new material row to the dataset with auto-assigned uid."""
    try:
        result = await service.add_material(dataset_id, body.model_dump(), db)
        logger.info("[ADD_ROW] dataset=%s formula=%s → uid=%d", dataset_id[:8], body.formula, result.uid)
        return result
    except ValueError as e:
        logger.error("[ADD_ROW] FAILED dataset=%s: %s", dataset_id[:8], e)
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/{dataset_id}/materials/{material_id}", response_model=MaterialResponse)
async def update_material(
    dataset_id: str,
    material_id: str,
    body: MaterialUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update a single material row."""
    # Only send non-None fields
    update_data = body.model_dump(exclude_unset=True)
    try:
        return await service.update_material(dataset_id, material_id, update_data, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/materials/bulk", response_model=BulkUpdateResponse)
async def bulk_update_materials(
    dataset_id: str,
    body: BulkMaterialRequest,
    db: AsyncSession = Depends(get_db),
):
    """Bulk update and/or delete materials."""
    updates = [{"id": u.id, "updates": u.updates} for u in body.updates]
    logger.info("[BULK] dataset=%s → %d updates, %d deletes", dataset_id[:8], len(updates), len(body.deletes))
    result = await service.bulk_update_materials(
        dataset_id, updates, body.deletes, db,
    )
    if result.errors:
        for err in result.errors:
            logger.warning("  ⚠ BULK error: %s", err)
    return result


@router.post("/{dataset_id}/materials/clear-column/{field_name}", response_model=BulkUpdateResponse)
async def clear_material_column(
    dataset_id: str,
    field_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Clear a column across all materials in a dataset (sets values to NULL / defaults).
    This is the S2 'delete column' remediation action in Review Issues.
    """
    try:
        return await service.clear_column(dataset_id, field_name, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{dataset_id}/materials/{material_id}", status_code=204)
async def delete_material(
    dataset_id: str,
    material_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a single material row (uid is NOT reassigned)."""
    try:
        await service.delete_material(dataset_id, material_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
