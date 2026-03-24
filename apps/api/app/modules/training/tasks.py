"""
Training task — runs as a FastAPI BackgroundTask (synchronous).
Uses its own sync DB session to write logs visible to the SSE stream.
"""
from piezo_ml.pipeline.trainer import TrainingPipeline
from packages.db.models.training import TrainingJob, TrainingLog, ModelArtifact
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import json

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '.env'))

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/piezo_ai")
# BackgroundTasks are sync, so we need a sync DB URL
sync_db_url = DB_URL.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
engine = create_engine(sync_db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

print(f"[TRAINING TASKS] DB URL resolved to: {sync_db_url.split('@')[0].split('://')[0]}://*****@{sync_db_url.split('@')[-1]}", flush=True)


def train_model_task(job_id: str, dataset_id: str, target: str, model_name: str, mode: str, params: dict, use_optuna: bool, optuna_trials: int):
    """Synchronous training function called by FastAPI BackgroundTasks."""
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            print(f"[TRAINING] Job {job_id} not found in DB, aborting.", flush=True)
            return
            
        job.status = "running"
        db.commit()
        print(f"[TRAINING] Job {job_id} marked as running", flush=True)

        # Step tracker
        step_counter = [0]

        def _log_callback(level: str, message: str, step_name: str, metadata: dict = None):
            step_counter[0] += 1
            print(f"👉 [PIEZO-ML] [{level}] STEP: {step_name} | MSG: {message}", flush=True)
            log_entry = TrainingLog(
                job_id=job_id,
                step=step_counter[0],
                level=level,
                message=message,
                metadata_json=metadata or {}
            )
            db.add(log_entry)
            db.commit()

        try:
            from packages.db.models.dataset import Material
            from piezo_ml.features.engineer import FeatureEngineer
            import numpy as np
            
            _log_callback("INFO", "Loading dataset from database", "load_data")
            materials = db.query(Material).filter(Material.dataset_id == dataset_id).all()
            
            engineer = FeatureEngineer()
            x_list, y_list = [], []
            for mat in materials:
                try:
                    vec, _ = engineer.compute_features(mat.formula)
                    tval = getattr(mat, target, None)
                    if tval is not None:
                        if target == "composite_d33":
                            wt = getattr(mat, "filler_wt_pct", 0) or 0
                            beta = getattr(mat, "beta_phase_pct", 0) or 0
                            
                            matrix = getattr(mat, "matrix_type", "")
                            matrix_enc = 1.0 if matrix == "pvdf" else (2.0 if matrix == "epoxy" else 0.0)
                            
                            morph = getattr(mat, "particle_morphology", "")
                            morph_enc = 1.0 if morph == "spherical" else (2.0 if morph == "rod" else 0.0)
                            
                            comp_features = np.array([wt, beta, matrix_enc, morph_enc])
                            vec = np.concatenate([vec, comp_features])
                            
                        x_list.append(vec)
                        y_list.append(tval)
                except Exception:
                    pass
                    
            if len(x_list) < 10:
                raise ValueError(f"Insufficient data points for training (got {len(x_list)}, need ≥10)")

            X = np.array(x_list)
            y = np.array(y_list)
            
            _log_callback("INFO", f"Dataset loaded: {len(x_list)} samples, {X.shape[1]} features", "data_ready")

            pipeline = TrainingPipeline(log_callback=_log_callback)
            
            if mode == "expert":
                result = pipeline.run_expert(job_id, X, y, model_name, params, target, use_optuna, optuna_trials)
            else:
                result = pipeline.run_expert(job_id, X, y, model_name, params, target, False, 0)
                
            job.status = "completed"
            
            metrics = result.get("metrics", {})
            if target == "d33":
                job.r2_d33 = metrics.get("r2_test")
                job.rmse_d33 = metrics.get("rmse_test")
            elif target == "tc":
                job.r2_tc = metrics.get("r2_test")
                job.rmse_tc = metrics.get("rmse_test")
            
            import uuid as uuid_mod
            alias = f"{model_name}-{target}-{uuid_mod.uuid4().hex[:4]}".upper()
            artifact = ModelArtifact(
                job_id=job_id,
                target=target,
                model_name=model_name,
                alias=alias,
                artifact_path=result["artifact_path"],
                is_best=False,
                r2_test=metrics.get("r2_test"),
                rmse_test=metrics.get("rmse_test"),
                r2_train=metrics.get("r2_train"),
                rmse_train=metrics.get("rmse_train"),
                hyperparameters=params
            )
            db.add(artifact)
            db.commit()
            
            r2_val = metrics.get('r2_test', 'N/A')
            rmse_val = metrics.get('rmse_test', 'N/A')
            r2_str = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val)
            rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else str(rmse_val)
            
            _log_callback("INFO", f"Training complete! R²={r2_str}, RMSE={rmse_str}", "complete")
            print(f"[TRAINING] Job {job_id} completed successfully", flush=True)
            
        except Exception as e:
            _log_callback("ERROR", f"Training failed: {str(e)}", "error")
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            print(f"[TRAINING] Job {job_id} failed: {e}", flush=True)
            
    finally:
        db.close()
