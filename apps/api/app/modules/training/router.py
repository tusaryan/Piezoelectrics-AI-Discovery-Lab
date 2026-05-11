"""
Training Router — API endpoints + WebSocket for real-time log streaming.
"""

from __future__ import annotations

import asyncio
import queue
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.training.schemas import (
    AlgorithmInfoResponse,
    DatasetValidationRequest,
    DatasetValidationResponse,
    TrainedModelResponse,
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingResultsResponse,
)
from app.modules.training.service import training_service

from piezo_db.models import TrainedModel, TrainingJob
from piezo_ml.models import get_algorithm_list

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /algorithms — list available algorithms + hyperparameter metadata
# ---------------------------------------------------------------------------
@router.get("/algorithms", response_model=list[AlgorithmInfoResponse])
async def list_algorithms():
    """Return all 8 supported algorithms with hyperparameter definitions."""
    return get_algorithm_list()


# ---------------------------------------------------------------------------
# POST /validate — pre-training dataset validation
# ---------------------------------------------------------------------------
@router.post("/validate", response_model=DatasetValidationResponse)
async def validate_for_training(
    request: DatasetValidationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Check dataset for missing/sentinel values before training."""
    try:
        result = await training_service.validate_dataset(
            db, str(request.dataset_id), request.selected_fields,
            targets=request.targets or None,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# POST /jobs — create a new training job
# ---------------------------------------------------------------------------
@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    body: TrainingJobCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create and start a new training job."""
    try:
        job = await training_service.create_job(db, body.model_dump())
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GET /jobs — list all training jobs
# ---------------------------------------------------------------------------
@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(db: AsyncSession = Depends(get_db)):
    """List all training jobs, most recent first."""
    result = await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.desc())
    )
    jobs = result.scalars().all()
    responses = []
    for job in jobs:
        resp = _job_to_response(job)
        # Enrich with live progress for active jobs
        active = training_service.get_active_job(str(job.id))
        if active:
            resp.progress_pct = active.progress
            resp.current_stage = active.current_stage
        responses.append(resp)
    return responses


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} — get job status with live progress
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get training job status, enriched with live progress if running."""
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    resp = _job_to_response(job)
    active = training_service.get_active_job(str(job_id))
    if active:
        resp.progress_pct = active.progress
        resp.current_stage = active.current_stage
        # If worker completed, persist results to DB
        if active.completed and active.results and job.status in ("queued", "running"):
            await _persist_completed_job(db, job, active)
            resp = _job_to_response(job)
        elif active.completed and active.error:
            if active.error == "cancelled":
                job.status = "cancelled"
            else:
                job.status = "failed"
                job.error_message = active.error
            job.completed_at = datetime.utcnow()
            await db.flush()
            resp = _job_to_response(job)
            training_service.cleanup_job(str(job_id))
    return resp


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/stop — cancel a running job
# ---------------------------------------------------------------------------
@router.post("/jobs/{job_id}/stop")
async def stop_training_job(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Cancel a running training job."""
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    stopped = training_service.stop_job(str(job_id))
    if stopped:
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        await db.flush()
        return {"status": "cancelled", "job_id": str(job_id)}
    else:
        raise HTTPException(status_code=400, detail="Job is not running or already finished")


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id} — delete a completed/failed job
# ---------------------------------------------------------------------------
@router.delete("/jobs/{job_id}", status_code=204)
async def delete_training_job(job_id: UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    await db.delete(job)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/results — trained models for a completed job
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/results", response_model=TrainingResultsResponse)
async def get_training_results(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get trained models and metrics for a completed job."""
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    result = await db.execute(
        select(TrainedModel).where(TrainedModel.training_job_id == job_id)
    )
    models = result.scalars().all()

    convergence = {}
    active = training_service.get_active_job(str(job_id))
    if active and active.convergence_data:
        convergence = active.convergence_data

    return TrainingResultsResponse(
        job_id=str(job_id),
        status=job.status,
        models=[_model_to_response(m) for m in models],
        convergence_data=convergence,
    )


# ---------------------------------------------------------------------------
# WebSocket /ws/{job_id} — real-time log + progress streaming
# ---------------------------------------------------------------------------
@router.websocket("/ws/{job_id}")
async def training_log_stream(websocket: WebSocket, job_id: str):
    """Stream training logs, progress, and convergence data in real-time."""
    await websocket.accept()
    try:
        active = training_service.get_active_job(job_id)
        if not active:
            await websocket.send_json({
                "type": "error",
                "message": "Job not found or not active",
            })
            await websocket.close()
            return

        while True:
            # Drain log queue
            msgs_sent = 0
            while msgs_sent < 50:  # batch limit per iteration
                try:
                    msg = active.log_queue.get_nowait()
                    await websocket.send_json(msg)
                    msgs_sent += 1
                except queue.Empty:
                    break

            # Check if job is done
            if active.completed:
                # Drain remaining
                while True:
                    try:
                        msg = active.log_queue.get_nowait()
                        await websocket.send_json(msg)
                    except queue.Empty:
                        break
                break

            await asyncio.sleep(0.1)

        # ── Persist completed job to DB immediately after WS loop ──
        # This ensures TrainedModel rows exist for the Predict section
        # without requiring a separate GET /jobs/{id} call.
        from app.core.database import async_session_factory
        try:
            async with async_session_factory() as db:
                job = await db.get(TrainingJob, job_id)
                if job and job.status in ("queued", "running"):
                    if active.completed and active.results:
                        await _persist_completed_job(db, job, active)
                        await db.commit()
                    elif active.completed and active.error:
                        if active.error == "cancelled":
                            job.status = "cancelled"
                        else:
                            job.status = "failed"
                            job.error_message = active.error
                        job.completed_at = datetime.utcnow()
                        await db.flush()
                        await db.commit()
                        training_service.cleanup_job(job_id)
        except Exception as persist_err:
            print(f"[Train WS] ⚠️ DB persist failed (will retry on GET): {persist_err}")

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_to_response(job: TrainingJob) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=str(job.id),
        dataset_id=str(job.dataset_id),
        status=job.status,
        mode=job.mode,
        targets=job.targets,
        algorithms=job.algorithms,
        hyperparameters=job.hyperparameters,
        selected_fields=job.selected_fields,
        progress_pct=job.progress_pct,
        current_stage=job.current_stage,
        initial_rows=job.initial_rows,
        initial_columns=job.initial_columns,
        final_rows=job.final_rows,
        final_columns=job.final_columns,
        artifact_dir=job.artifact_dir,
        error_message=job.error_message,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        created_at=job.created_at.isoformat() if job.created_at else "",
    )


def _model_to_response(m: TrainedModel) -> TrainedModelResponse:
    return TrainedModelResponse(
        id=str(m.id),
        display_name=m.display_name,
        target=m.target,
        algorithm=m.algorithm,
        r2_score=m.r2_score,
        rmse=m.rmse,
        hyperparameters=m.hyperparameters,
        feature_dim=m.feature_dim,
        n_train_samples=m.n_train_samples,
        n_test_samples=m.n_test_samples,
        model_file_path=m.model_file_path,
        training_duration_s=m.training_duration_s,
        is_default=m.is_default,
        created_at=m.created_at.isoformat() if m.created_at else "",
    )


async def _persist_completed_job(
    db: AsyncSession, job: TrainingJob, active,
) -> None:
    """Persist training results to DB after worker completes."""
    results = active.results
    job.status = "completed"
    job.started_at = job.started_at or job.created_at
    job.completed_at = datetime.utcnow()
    job.progress_pct = 100.0
    job.current_stage = "Complete"
    job.initial_rows = results.get("initial_rows")
    job.initial_columns = results.get("initial_columns")
    job.final_rows = results.get("final_rows")
    job.final_columns = results.get("final_columns")
    job.artifact_dir = results.get("artifact_dir")

    for model_data in results.get("models", []):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        display = f"{model_data['algorithm']}_{model_data['target']}_{ts}"
        tm = TrainedModel(
            display_name=display,
            training_job_id=job.id,
            dataset_id=job.dataset_id,
            target=model_data["target"],
            algorithm=model_data["algorithm"],
            r2_score=model_data["r2"],
            rmse=model_data["rmse"],
            hyperparameters=model_data["hyperparameters"],
            feature_version="v4",
            feature_dim=model_data.get("feature_dim", 0),
            n_train_samples=model_data["n_train"],
            n_test_samples=model_data["n_test"],
            supported_elements=model_data.get("supported_elements", []),
            model_file_path=model_data.get("model_path", ""),
            artifact_dir=model_data.get("artifact_dir", ""),
            training_duration_s=model_data["training_duration_s"],
        )
        db.add(tm)

    await db.flush()
    training_service.cleanup_job(str(job.id))
