import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from packages.db.models.training import TrainingJob, TrainingLog
from apps.api.app.modules.training.schemas import StartTrainingRequest

class JobManager:
    @staticmethod
    async def create_job(db: AsyncSession, request: StartTrainingRequest) -> TrainingJob:
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            dataset_id=request.dataset_id,
            name=f"Training {request.model_name} on {request.target}",
            mode=request.mode,
            status="queued",
            config={
                "target": request.target,
                "model_name": request.model_name,
                "params": request.params,
                "use_optuna": getattr(request, "use_optuna", False),
                "optuna_trials": getattr(request, "optuna_trials", 0)
            }
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job

    @staticmethod
    def dispatch_training_task(background_tasks, job_id: str, request: StartTrainingRequest):
        """Dispatch training as a FastAPI BackgroundTask (no Celery/Redis)."""
        from apps.api.app.modules.training.tasks import train_model_task

        print(f"\n{'='*60}", flush=True)
        print(f"[JOB_MANAGER] Dispatching task via BackgroundTasks...", flush=True)
        print(f"  Job ID:    {job_id}", flush=True)
        print(f"  Target:    {request.target}", flush=True)
        print(f"  Model:     {request.model_name}", flush=True)
        print(f"  Dataset:   {request.dataset_id}", flush=True)
        print(f"{'='*60}", flush=True)
        
        background_tasks.add_task(
            train_model_task,
            job_id=job_id,
            dataset_id=request.dataset_id,
            target=request.target,
            model_name=request.model_name,
            mode=request.mode,
            params=request.params,
            use_optuna=request.use_optuna,
            optuna_trials=request.optuna_trials
        )
        print(f"[JOB_MANAGER] -> Task added to BackgroundTasks queue", flush=True)

    @staticmethod
    async def get_job_status(db: AsyncSession, job_id: str) -> Optional[TrainingJob]:
        result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        return result.scalars().first()
