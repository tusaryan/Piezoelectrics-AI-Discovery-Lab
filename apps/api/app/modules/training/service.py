"""
Training Service — job management, background worker, and cancel propagation.

Uses threading.Thread (not multiprocessing) to run ML pipeline.
Cancel signal via threading.Event shared with the ML orchestrator.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from piezo_db.models import Dataset, Material, TrainedModel, TrainingJob

from piezo_ml.models.trainer import TrainingCancelledError
from piezo_ml.pipeline.sentinel_handler import detect_sentinel_issues, get_default_strategies
from piezo_ml.pipeline.training_orchestrator import TrainingConfig, TrainingOrchestrator


@dataclass
class ActiveJob:
    """In-memory state for a running training job."""
    job_id: str
    thread: threading.Thread
    cancel_event: threading.Event
    log_queue: queue.Queue
    progress: float = 0.0
    current_stage: str = ""
    convergence_data: dict[str, list[dict]] = field(default_factory=dict)
    results: dict[str, Any] | None = None
    error: str | None = None
    completed: bool = False


class TrainingService:
    """Manages training jobs: creation, execution, cancellation."""

    def __init__(self) -> None:
        self._active_jobs: dict[str, ActiveJob] = {}

    # ---- dataset loading ----

    async def _load_dataset_df(
        self, db: AsyncSession, dataset_id: str, selected_fields: list[str],
    ) -> pd.DataFrame:
        """Load material rows from DB into a DataFrame."""
        result = await db.execute(
            select(Material)
            .where(Material.dataset_id == dataset_id)
            .order_by(Material.uid)
        )
        materials = result.scalars().all()
        if not materials:
            raise ValueError(f"No materials found for dataset {dataset_id}")

        # Build rows with only selected fields + uid + formula
        all_fields = {"uid", "formula"} | set(selected_fields)
        rows = []
        for m in materials:
            row: dict[str, Any] = {}
            for f in all_fields:
                row[f] = getattr(m, f, None)
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- validation ----

    async def validate_dataset(
        self, db: AsyncSession, dataset_id: str, selected_fields: list[str],
    ) -> dict:
        """Pre-training validation — detect sentinel values and missing data."""
        df = await self._load_dataset_df(db, dataset_id, selected_fields)
        issues = detect_sentinel_issues(df, selected_fields)
        defaults = get_default_strategies(issues)
        return {
            "dataset_id": str(dataset_id),
            "total_rows": len(df),
            "issues": [
                {
                    "field": i.field, "issue_type": i.issue_type,
                    "count": i.count, "total": i.total,
                    "message": i.message, "suggestion": i.suggestion,
                    "default_strategy": i.default_strategy,
                    "allowed_strategies": i.allowed_strategies,
                }
                for i in issues
            ],
            "default_strategies": defaults,
        }

    # ---- job creation ----

    async def create_job(
        self, db: AsyncSession, config: dict,
    ) -> TrainingJob:
        """Create a training job in DB and start background worker."""
        dataset_id = str(config["dataset_id"])

        # Verify dataset exists and is ready
        ds = await db.get(Dataset, config["dataset_id"])
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        if ds.status != "ready":
            raise ValueError(f"Dataset {dataset_id} is not ready (status: {ds.status})")

        job = TrainingJob(
            dataset_id=config["dataset_id"],
            status="queued",
            mode=config.get("mode", "manual"),
            targets=config["targets"],
            algorithms=config["algorithms"],
            hyperparameters=config.get("hyperparameters"),
            selected_fields=config["selected_fields"],
        )
        db.add(job)
        await db.flush()

        job_id = str(job.id)

        # Load dataset into DataFrame before spawning thread
        df = await self._load_dataset_df(db, dataset_id, config["selected_fields"])

        # Set up background worker
        cancel_event = threading.Event()
        log_q: queue.Queue = queue.Queue(maxsize=5000)

        training_config = TrainingConfig(
            dataset_id=dataset_id,
            training_id=job_id,
            targets=config["targets"],
            algorithms=config["algorithms"],
            hyperparameters=config.get("hyperparameters", {}),
            selected_fields=config["selected_fields"],
            missing_strategies=config.get("missing_strategies", {}),
            mode=config.get("mode", "manual"),
        )

        active = ActiveJob(
            job_id=job_id, thread=None,  # type: ignore
            cancel_event=cancel_event, log_queue=log_q,
        )
        self._active_jobs[job_id] = active

        t = threading.Thread(
            target=self._run_worker,
            args=(job_id, df, training_config, cancel_event, log_q),
            daemon=True,
        )
        active.thread = t
        t.start()

        return job

    # ---- background worker ----

    def _run_worker(
        self,
        job_id: str,
        df: pd.DataFrame,
        config: TrainingConfig,
        cancel_event: threading.Event,
        log_q: queue.Queue,
    ) -> None:
        """Background thread that runs the ML pipeline."""
        active = self._active_jobs.get(job_id)
        if not active:
            return

        def log_callback(level: str, msg: str):
            try:
                log_q.put_nowait({
                    "type": "log", "level": level, "message": msg,
                    "timestamp": datetime.now().isoformat(),
                })
            except queue.Full:
                pass
            # Also print to backend terminal
            prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(level, "")
            print(f"[Train {job_id[:8]}] {prefix} {msg}")

        def progress_callback(pct: float, stage: str):
            active.progress = pct
            active.current_stage = stage
            try:
                log_q.put_nowait({
                    "type": "progress", "pct": pct, "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                })
            except queue.Full:
                pass

        def convergence_callback(target: str, iteration: int, metric: float):
            active.convergence_data.setdefault(target, []).append(
                {"iteration": iteration, "metric": metric}
            )
            try:
                log_q.put_nowait({
                    "type": "convergence", "target": target,
                    "iteration": iteration, "metric": metric,
                })
            except queue.Full:
                pass

        try:
            orchestrator = TrainingOrchestrator(
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                log_callback=log_callback,
                convergence_callback=convergence_callback,
            )
            output = orchestrator.run(df, config)

            # Store results for DB persistence
            active.results = {
                "training_id": output.training_id,
                "artifact_dir": output.artifact_dir,
                "initial_rows": output.initial_rows,
                "initial_columns": output.initial_columns,
                "final_rows": output.final_rows,
                "final_columns": output.final_columns,
                "models": [
                    {
                        "target": r.target,
                        "algorithm": r.algorithm,
                        "r2": r.r2,
                        "rmse": r.rmse,
                        "hyperparameters": r.hyperparameters,
                        "n_train": r.n_train,
                        "n_test": r.n_test,
                        "training_duration_s": r.training_duration_s,
                        "convergence_data": r.convergence_data,
                        "feature_importances": r.feature_importances,
                        "model_path": str(ma.model_path),
                        "feature_dim": ma.metadata.get("feature_dim", 0),
                        "supported_elements": ma.metadata.get("supported_elements", []),
                        "artifact_dir": ma.metadata.get("source_artifact_dir", ""),
                    }
                    for r, ma in zip(output.results, output.model_artifacts)
                ],
            }
            active.completed = True
            try:
                log_q.put_nowait({"type": "complete", "results": active.results})
            except queue.Full:
                pass

        except TrainingCancelledError:
            active.error = "cancelled"
            active.completed = True
            try:
                log_q.put_nowait({"type": "cancelled", "message": "Training cancelled by user"})
            except queue.Full:
                pass

        except Exception as e:
            active.error = str(e)
            active.completed = True
            log_callback("error", f"Training failed: {e}")
            try:
                log_q.put_nowait({"type": "error", "message": str(e)})
            except queue.Full:
                pass

    # ---- stop ----

    def stop_job(self, job_id: str) -> bool:
        """Signal a running job to cancel."""
        active = self._active_jobs.get(job_id)
        if active and not active.completed:
            active.cancel_event.set()
            return True
        return False

    # ---- status ----

    def get_active_job(self, job_id: str) -> ActiveJob | None:
        return self._active_jobs.get(job_id)

    def cleanup_job(self, job_id: str) -> None:
        self._active_jobs.pop(job_id, None)


# Singleton service instance
training_service = TrainingService()
