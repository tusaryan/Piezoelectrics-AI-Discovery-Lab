from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Any, Dict
from pydantic import BaseModel

from apps.api.app.core.database import get_db
from apps.api.app.modules.dataset.schemas import (
    StandardResponse, DatasetListResponse, DatasetDetailResponse,
    MaterialListResponse, MaterialUpdate, ResolveIssueRequest, DatasetIssuesResponse
)
from apps.api.app.modules.dataset.service import DatasetService

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

class ConfirmMappingRequest(BaseModel):
    mapping: Dict[str, Optional[str]]  # {internal_field: csv_column_name_or_null}

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Step 1: Upload file → get column suggestions for SchemaMapper UI."""
    import traceback
    try:
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV and XLSX files allowed")
            
        contents = await file.read()
        result = await DatasetService.process_upload_step1(db, contents, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        with open("/tmp/fastapi_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"data": result, "meta": {}}

@router.post("/{dataset_id}/confirm-mapping")
async def confirm_column_mapping(
    dataset_id: str,
    request: ConfirmMappingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Step 2: Apply user-confirmed column mapping → validate → return issues or processing dataset."""
    import traceback
    try:
        # Filter out None/empty mappings
        mapping = {k: v for k, v in request.mapping.items() if v}
        dataset, issues = await DatasetService.process_upload_step2(db, dataset_id, mapping)
        
        if dataset.status == "processing":
            background_tasks.add_task(DatasetService._run_processing_task, dataset_id, {})
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        with open("/tmp/fastapi_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    issue_list = [
        {
            "row_idx": i.row_idx, "column": i.column, "issue_type": i.issue_type,
            "severity": i.severity, "description": i.description, 
            "auto_fixable": i.auto_fixable, "choices": i.choices
        } for i in issues
    ]
    
    return {
        "data": {
            "id": str(dataset.id),
            "name": dataset.name,
            "status": dataset.status,
            "row_count": dataset.row_count,
            "has_d33": dataset.has_d33,
            "has_tc": dataset.has_tc,
            "column_map": dataset.column_map,
            "created_at": str(dataset.created_at) if dataset.created_at else None,
            "issues": issue_list
        },
        "meta": {"issue_count": len(issues)}
    }


@router.get("", response_model=DatasetListResponse)
async def list_datasets(db: AsyncSession = Depends(get_db)):
    datasets = await DatasetService.list_datasets(db)
    return {
        "data": [
            {
                "id": str(d.id), "name": d.name, "status": d.status,
                "row_count": d.row_count, "has_d33": d.has_d33, "has_tc": d.has_tc,
                "column_map": d.column_map,
                "created_at": str(d.created_at) if d.created_at else None
            } for d in datasets
        ],
        "meta": {"total": len(datasets)}
    }

from fastapi.responses import StreamingResponse
import asyncio
from apps.api.app.modules.dataset.service import DATASET_PROGRESS

@router.get("/{dataset_id}/progress")
async def stream_dataset_progress(dataset_id: str):
    """Server-Sent Events endpoint strictly pushing exact task progress (0-100)."""
    async def event_generator():
        last_progress = -1
        while True:
            current = DATASET_PROGRESS.get(dataset_id, 0)
            if current != last_progress:
                yield f"data: {current}\n\n"
                last_progress = current
                
            if current == 100 or isinstance(current, str):
                break
                
            await asyncio.sleep(0.1) # Check loop is lightweight

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await DatasetService.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    return {
        "data": {
            "id": str(dataset.id), "name": dataset.name, "status": dataset.status,
            "row_count": dataset.row_count, "has_d33": dataset.has_d33, "has_tc": dataset.has_tc,
            "column_map": dataset.column_map,
            "created_at": str(dataset.created_at) if dataset.created_at else None,
            "issues": [] # would fetch from a separate table if fully persisted
        },
        "meta": {}
    }

@router.get("/{dataset_id}/materials", response_model=MaterialListResponse)
async def get_materials(
    dataset_id: str, 
    page: int = Query(1, ge=1), 
    per_page: int = Query(50, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    skip = (page - 1) * per_page
    materials, total = await DatasetService.get_materials(db, dataset_id, skip, per_page)
    
    return {
        "data": [
            {
                "id": str(m.id), "dataset_id": str(m.dataset_id), "formula": m.formula,
                "sintering_temp": m.sintering_temp, "d33": m.d33, "tc": m.tc,
                "is_imputed": m.is_imputed, "is_tc_ai_generated": m.is_tc_ai_generated
            } for m in materials
        ],
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": max(1, (total + per_page - 1) // per_page)
        }
    }

@router.put("/{dataset_id}/materials/{material_id}")
async def update_material(
    dataset_id: str,
    material_id: str, 
    update: MaterialUpdate, 
    db: AsyncSession = Depends(get_db)
):
    mat = await DatasetService.update_material(db, material_id, update.model_dump(exclude_unset=True))
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    return {"data": mat, "meta": {}}

@router.get("/{dataset_id}/issues", response_model=DatasetIssuesResponse)
async def get_dataset_issues(dataset_id: str):
    issues = await DatasetService.get_dataset_issues(dataset_id)
    issue_list = [
        {
            "row_idx": i.row_idx, "column": i.column, "issue_type": i.issue_type,
            "severity": i.severity, "description": i.description, 
            "auto_fixable": i.auto_fixable, "choices": i.choices
        } for i in issues
    ]
    return {"data": issue_list, "meta": {"issue_count": len(issues)}}

@router.post("/{dataset_id}/resolve-issue", response_model=DatasetDetailResponse)
async def resolve_dataset_issue(
    dataset_id: str,
    request: ResolveIssueRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    success, remaining_issues, dataset = await DatasetService.resolve_dataset_issues(
        db, dataset_id, request.resolutions
    )
    
    if success and dataset and dataset.status == "processing":
        background_tasks.add_task(DatasetService._run_processing_task, dataset_id, request.resolutions)
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found or raw file missing")
        
    issue_list = [
        {
            "row_idx": i.row_idx, "column": i.column, "issue_type": i.issue_type,
            "severity": i.severity, "description": i.description, 
            "auto_fixable": i.auto_fixable, "choices": i.choices
        } for i in remaining_issues
    ]
    
    return {
        "success": True,
        "data": {
            "id": str(dataset.id),
            "name": dataset.name,
            "status": dataset.status,
            "row_count": dataset.row_count,
            "has_d33": dataset.has_d33,
            "has_tc": dataset.has_tc,
            "created_at": str(dataset.created_at) if dataset.created_at else None,
            "issues": issue_list
        },
        "meta": {"remaining_issues": len(remaining_issues)}
    }

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a dataset and all associated materials."""
    success = await DatasetService.delete_dataset(db, dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True, "data": None, "meta": {}}
