from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
import uuid
import asyncio
import json

from piezo_ml.optimization.nsga2_optimizer import ParetoOptimizer
from apps.api.app.core.database import get_db, AsyncSessionLocal
from packages.db.models.prediction import ParetoRun, ParetoSolution
from apps.api.app.modules.inverse.schemas import ParetoOptimizerConfig, ParetoRunResponse, ParetoSolutionResponse

router = APIRouter(prefix="/api/v1/inverse", tags=["Inverse Design"])

_PARETO_STREAMS: Dict[str, asyncio.Queue] = {}

async def run_pareto_background_job(job_id: str, config_dict: dict):
    from apps.api.app.core.config import settings
    from piezo_ml.features.engineer import FeatureEngineer
    import os
    import joblib

    engineer = FeatureEngineer()
    d33_model = None
    tc_model = None
    
    d33_path = os.path.join(settings.model_artifacts_path, "active_d33_model.pkl")
    tc_path = os.path.join(settings.model_artifacts_path, "active_tc_model.pkl")
    
    if os.path.exists(d33_path):
        d33_model = joblib.load(d33_path)
    if os.path.exists(tc_path):
        tc_model = joblib.load(tc_path)
        
    optimizer = ParetoOptimizer(config=config_dict, models=(d33_model, tc_model), engineer=engineer)
    queue = _PARETO_STREAMS.get(job_id)
    
    async def stream_cb(msg: dict):
        if queue:
            await queue.put(msg)
            
    try:
        solutions = await optimizer.run_optimization_async(cb=stream_cb)
        
        async with AsyncSessionLocal() as session:
            # Add solutions
            db_solutions = []
            for sol in solutions:
                comp_dict = {f"var_{i}": v for i, v in enumerate(sol["composition"])}
                db_sol = ParetoSolution(
                    id=uuid.uuid4(),
                    run_id=UUID(job_id),
                    composition=comp_dict,
                    predicted_d33=sol["predicted_d33"],
                    predicted_tc=sol["predicted_tc"],
                    predicted_hardness=sol["predicted_hardness"],
                    rank=sol.get("rank", 1),
                    crowd_distance=sol.get("crowd_distance", 0.0),
                    use_case=sol.get("use_case", "")
                )
                db_solutions.append(db_sol)
                session.add(db_sol)
                
            # Update run
            stmt = select(ParetoRun).where(ParetoRun.id == UUID(job_id))
            result = await session.execute(stmt)
            db_run = result.scalar_one_or_none()
            if db_run:
                db_run.status = "success"
                db_run.result_count = len(db_solutions)
                
            await session.commit()
            
        if queue:
            await queue.put({"type": "done", "result_count": len(solutions)})
            
    except Exception as e:
        if queue:
            await queue.put({"type": "error", "message": str(e)})
            
        async with AsyncSessionLocal() as session:
            stmt = select(ParetoRun).where(ParetoRun.id == UUID(job_id))
            result = await session.execute(stmt)
            db_run = result.scalar_one_or_none()
            if db_run:
                db_run.status = "failed"
                await session.commit()
    finally:
        await asyncio.sleep(5)
        if job_id in _PARETO_STREAMS:
            del _PARETO_STREAMS[job_id]


@router.post("/pareto/start")
async def start_pareto_optimization(
    config: ParetoOptimizerConfig,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    obj_dict = {obj: True for obj in config.objectives}
    new_run = ParetoRun(
        id=uuid.uuid4(),
        name=config.name,
        algorithm=config.algorithm,
        objectives=obj_dict,
        constraints=config.constraints,
        n_generations=config.n_generations,
        population_size=config.population_size,
        status="running"
    )
    db.add(new_run)
    await db.commit()
    await db.refresh(new_run)
    
    job_id_str = str(new_run.id)
    _PARETO_STREAMS[job_id_str] = asyncio.Queue()
    
    background_tasks.add_task(run_pareto_background_job, job_id_str, config.model_dump())
    return {"success": True, "data": {"run_id": job_id_str}, "error": None}


@router.get("/pareto/{id}")
async def get_pareto_run(id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(ParetoRun).where(ParetoRun.id == id)
    result = await db.execute(stmt)
    db_run = result.scalar_one_or_none()
    if not db_run:
        return {"success": False, "data": None, "error": {"code": "NOT_FOUND", "message": "Pareto run not found"}}
    return {"success": True, "data": ParetoRunResponse.model_validate(db_run).model_dump(mode="json"), "error": None}


@router.get("/pareto/{id}/front")
async def get_pareto_front(id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(ParetoSolution).where(ParetoSolution.run_id == id)
    result = await db.execute(stmt)
    solutions = result.scalars().all()
    data = [ParetoSolutionResponse.model_validate(s).model_dump(mode="json") for s in solutions]
    return {"success": True, "data": data, "error": None}


@router.get("/pareto/{id}/convergence")
async def stream_pareto_convergence(id: str):
    queue = _PARETO_STREAMS.get(id)
    if not queue:
        async def empty_stream():
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")
        
    async def event_generator():
        while True:
            try:
                msg = await queue.get()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ["done", "error"]:
                    break
            except asyncio.CancelledError:
                break
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")
