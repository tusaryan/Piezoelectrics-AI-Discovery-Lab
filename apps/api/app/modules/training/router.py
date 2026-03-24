import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import shutil
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, cast, String

from apps.api.app.core.database import get_db
from apps.api.app.modules.training.schemas import StartTrainingRequest, TrainingJobResponse
from apps.api.app.modules.training.job_manager import JobManager
from packages.db.models.training import TrainingLog, TrainingJob, ModelArtifact
from piezo_ml.models.registry import get_model_schema

router = APIRouter(prefix="/api/v1/training", tags=["training"])

@router.get("/model-schema")
async def get_schema():
    return {"data": get_model_schema(), "meta": {}}

@router.get("/artifacts/{dataset_id}")
async def get_dataset_artifacts(dataset_id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(ModelArtifact)
        .join(TrainingJob, ModelArtifact.job_id == TrainingJob.id)
        .where(cast(TrainingJob.dataset_id, String) == dataset_id, TrainingJob.status == "completed")
    )
    result = await db.execute(stmt)
    artifacts = result.scalars().all()
    
    data = []
    for art in artifacts:
        data.append({
            "id": str(art.id),
            "target": art.target,
            "model_name": art.model_name,
            "r2_test": art.r2_test,
            "rmse_test": art.rmse_test,
            "job_id": str(art.job_id),
            "created_at": art.created_at.isoformat(),
            "alias": art.alias or art.model_name
        })
    return {"data": data}

@router.get("/models")
async def get_all_models(db: AsyncSession = Depends(get_db)):
    stmt = select(ModelArtifact).order_by(ModelArtifact.created_at.desc())
    result = await db.execute(stmt)
    artifacts = result.scalars().all()
    
    data = []
    for art in artifacts:
        data.append({
            "id": str(art.id),
            "target": art.target,
            "model_name": art.model_name,
            "r2_test": art.r2_test,
            "rmse_test": art.rmse_test,
            "job_id": str(art.job_id),
            "created_at": art.created_at.isoformat()
        })
    return {"data": data}

@router.get("/available-targets")
async def get_available_targets(db: AsyncSession = Depends(get_db)):
    """Return which prediction targets have trained models available."""
    stmt = select(ModelArtifact.target).distinct()
    result = await db.execute(stmt)
    trained_targets = [row[0] for row in result.all()]
    
    # Also check for active model files on disk
    from apps.api.app.core.config import settings
    active_models = {}
    for target in ["d33", "tc", "hardness", "composite_d33"]:
        model_path = os.path.join(settings.model_artifacts_path, f"active_{target}_model.pkl")
        active_models[target] = os.path.exists(model_path)
    
    return {
        "data": {
            "trained_targets": trained_targets,
            "active_models": active_models
        }
    }

@router.delete("/models/{model_id}")
async def delete_model(model_id: str, db: AsyncSession = Depends(get_db)):
    art = await db.get(ModelArtifact, model_id)
    if not art:
        raise HTTPException(status_code=404, detail="Model artifact not found")
        
    if art.artifact_path and os.path.exists(art.artifact_path):
        try:
            os.remove(art.artifact_path)
        except Exception as e:
            print(f"Failed to delete model file {art.artifact_path}: {e}")
            
    await db.delete(art)
    await db.commit()
    return {"message": "Model deleted successfully"}

@router.post("/models/{model_id}/activate")
async def activate_model(model_id: str, db: AsyncSession = Depends(get_db)):
    art = await db.get(ModelArtifact, model_id)
    if not art:
        raise HTTPException(status_code=404, detail="Model artifact not found")
        
    if not art.artifact_path or not os.path.exists(art.artifact_path):
        raise HTTPException(status_code=404, detail="Model file missing on disk")
        
    from apps.api.app.core.config import settings
    
    target_clean = art.target.lower().replace(" ", "_")
    dest_filename = f"active_{target_clean}_model.pkl"
    dest_path = os.path.join(settings.model_artifacts_path, dest_filename)
    
    try:
        shutil.copy2(art.artifact_path, dest_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy model: {e}")
        
    return {"message": f"Model activated for {art.target}"}


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(
    request: StartTrainingRequest, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    job = await JobManager.create_job(db, request)
    
    try:
        print(f"[API] Dispatching job {job.id} via BackgroundTasks...", flush=True)
        JobManager.dispatch_training_task(background_tasks, str(job.id), request)
        print(f"[API] -> Successfully dispatched {job.id}!", flush=True)
    except Exception as e:
        print(f"[API ERROR] Failed to dispatch task: {e}", flush=True)
        job.status = "failed"
        job.error_message = f"Failed to dispatch task: {str(e)}"
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "job_id": str(job.id),
        "status": job.status,
        "created_at": job.created_at,
        "message": "Training job started successfully"
    }

@router.post("/{job_id}/cancel")
async def cancel_training(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await JobManager.get_job_status(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status in ["completed", "failed", "cancelled"]:
        return {"message": f"Job already {job.status}"}
        
    # Mark as cancelled in DB (BackgroundTasks can't be revoked, but the
    # task checks job status and can exit early if cancelled)
    job.status = "cancelled"
    await db.commit()
    
    return {"message": "Job marked as cancelled"}

from datetime import datetime, timezone

async def log_generator(job_id: str, db: AsyncSession):
    last_seen_ts = datetime(2000, 1, 1)
    poll_interval = 0.5
    
    while True:
        await db.rollback()
        
        result = await db.execute(
            select(TrainingLog)
            .where(TrainingLog.job_id == job_id)
            .where(TrainingLog.timestamp > last_seen_ts)
            .order_by(TrainingLog.timestamp.asc())
        )
        logs = result.scalars().all()
        
        for log in logs:
            data = {
                "level": log.level,
                "message": log.message,
                "step": log.step,
                "timestamp": log.timestamp.isoformat(),
                "metadata": log.metadata_json
            }
            yield f"data: {json.dumps(data)}\n\n"
            last_seen_ts = log.timestamp
                
        await db.rollback()
        job = await JobManager.get_job_status(db, job_id)
        if job and job.status in ["completed", "failed", "cancelled"]:
            await asyncio.sleep(0.3)
            await db.rollback()
            result = await db.execute(
                select(TrainingLog)
                .where(TrainingLog.job_id == job_id)
                .where(TrainingLog.timestamp > last_seen_ts)
                .order_by(TrainingLog.timestamp.asc())
            )
            final_logs = result.scalars().all()
            for log in final_logs:
                data = {
                    "level": log.level,
                    "message": log.message,
                    "step": log.step,
                    "timestamp": log.timestamp.isoformat(),
                    "metadata": log.metadata_json
                }
                yield f"data: {json.dumps(data)}\n\n"
                
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            break
            
        await asyncio.sleep(poll_interval)

@router.get("/logs/{job_id}/stream")
async def stream_training_logs(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await JobManager.get_job_status(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return StreamingResponse(
        log_generator(job_id, db), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
