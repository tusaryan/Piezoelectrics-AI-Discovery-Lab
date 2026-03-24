from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any, Optional
from uuid import UUID
import uuid
import asyncio
import json
import time

from apps.api.app.core.database import get_db, AsyncSessionLocal
from packages.db.models.prediction import ActiveLearningRun
from piezo_ml.optimization.active_learning import ActiveLearningSimulator
import random
import math

router = APIRouter(prefix="/api/v1/active-learning", tags=["Active Learning"])

class ActiveLearningStartRequest(BaseModel):
    strategy: str = "UCB"
    acquisition_fn: str = "UCB"
    total_budget: int = 50
    n_simulations: int = 1
    
    model_config = ConfigDict(extra="ignore")

class ActiveLearningRunResponse(BaseModel):
    id: UUID
    strategy: str
    acquisition_fn: Optional[str]
    total_budget: int
    current_iteration: int
    found_max_d33_at_iter: Optional[int]
    final_max_d33: Optional[float]
    efficiency_vs_random: Optional[float]
    status: str
    
    model_config = ConfigDict(from_attributes=True)

_AL_STREAMS: Dict[str, asyncio.Queue] = {}

async def run_al_background_job(job_id: str, request: ActiveLearningStartRequest):
    queue = _AL_STREAMS.get(job_id)
    from apps.api.app.core.config import settings
    import os
    import joblib
    
    d33_model = None
    d33_path = os.path.join(settings.model_artifacts_path, "active_d33_model.pkl")
    if os.path.exists(d33_path):
        d33_model = joblib.load(d33_path)
    
    simulator = ActiveLearningSimulator(
        total_budget=request.total_budget,
        strategy=request.strategy,
        acquisition_fn=request.acquisition_fn,
        oracle_model=d33_model
    )
    
    async def stream_cb(msg: dict):
        if queue:
            await queue.put(msg)
            
        # Also update DB current iteration
        if msg.get("type") == "progress":
            async with AsyncSessionLocal() as session:
                stmt = select(ActiveLearningRun).where(ActiveLearningRun.id == UUID(job_id))
                res = await session.execute(stmt)
                db_run = res.scalar_one_or_none()
                if db_run:
                    db_run.current_iteration = msg.get("step", 0)
                    await session.commit()
            
    try:
        final_result = await simulator.simulate_async(cb=stream_cb)
        
        # Save results to DB
        async with AsyncSessionLocal() as session:
            stmt = select(ActiveLearningRun).where(ActiveLearningRun.id == UUID(job_id))
            res = await session.execute(stmt)
            db_run = res.scalar_one_or_none()
            if db_run:
                db_run.status = "success"
                db_run.found_max_d33_at_iter = final_result["iterations_to_max"]["strategy"]
                db_run.final_max_d33 = final_result["final_max_d33"]
                db_run.efficiency_vs_random = final_result["efficiency_gain"]
                # We could save curves in JSONB if we add a column, otherwise just stream them
                # For Phase 4, we store summary metrics. We'll pass the curve to frontend via stream final step.
                await session.commit()
                
        if queue:
            await queue.put({"type": "done", "result": final_result})
            
    except Exception as e:
        async with AsyncSessionLocal() as session:
            stmt = select(ActiveLearningRun).where(ActiveLearningRun.id == UUID(job_id))
            res = await session.execute(stmt)
            db_run = res.scalar_one_or_none()
            if db_run:
                db_run.status = "failed"
                await session.commit()
                
        if queue:
            await queue.put({"type": "error", "message": str(e)})
            
    finally:
        await asyncio.sleep(5)
        if job_id in _AL_STREAMS:
            del _AL_STREAMS[job_id]


@router.post("/start")
async def start_active_learning(
    request: ActiveLearningStartRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    new_run = ActiveLearningRun(
        id=uuid.uuid4(),
        strategy=request.strategy,
        acquisition_fn=request.acquisition_fn,
        total_budget=request.total_budget,
        current_iteration=0,
        status="running"
    )
    db.add(new_run)
    await db.commit()
    await db.refresh(new_run)
    
    job_id_str = str(new_run.id)
    _AL_STREAMS[job_id_str] = asyncio.Queue()
    
    background_tasks.add_task(run_al_background_job, job_id_str, request)
    
    return {"success": True, "data": {"job_id": job_id_str}, "error": None}


@router.get("/pareto")
async def get_pareto_front():
    """
    Returns a simulated 3D Pareto front (NSGA-II) balancing d33, Tc, and Vickers Hardness.
    """
    points = []
    # Generate a parametric 3D surface representing a trade-off curve
    # High d33 -> Low Tc, Low Hardness
    # High Tc -> Low d33, High Hardness
    # High Hardness -> Med d33, High Tc
    for i in range(120):
        # Parametric generation
        t = random.uniform(0, 1)
        u = random.uniform(0, 1)
        
        # Base trade-offs
        d33 = 100 + 500 * (1 - t) * math.sqrt(u)
        tc = 150 + 400 * t * random.uniform(0.8, 1.2)
        hardness = 300 + 800 * (t ** 2) * (1 - u)
        
        # Add noise
        d33 += random.uniform(-20, 20)
        tc += random.uniform(-10, 10)
        hardness += random.uniform(-50, 50)
        
        d33 = float(max(50.0, float(d33)))
        tc = float(max(100.0, float(tc)))
        hardness = float(max(150.0, float(hardness)))
        
        points.append({
            "id": f"mat_{i}",
            "d33": round(d33, 1),
            "tc": round(tc, 1),
            "hardness": round(hardness, 1),
            "formula": f"Material Candidate {i}"
        })
        
    return {"success": True, "data": points, "error": None}


@router.get("/{id}")
async def get_active_learning_run(id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(ActiveLearningRun).where(ActiveLearningRun.id == id)
    result = await db.execute(stmt)
    db_run = result.scalar_one_or_none()
    
    if not db_run:
        return {"success": False, "data": None, "error": {"code": "NOT_FOUND", "message": "Run not found"}}
        
    return {"success": True, "data": ActiveLearningRunResponse.model_validate(db_run).model_dump(mode="json"), "error": None}


@router.get("/{id}/stream")
async def stream_active_learning(id: str):
    queue = _AL_STREAMS.get(id)
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

# The /efficiency endpoint will get the data from the DB or completed simulation result if we persist it.
# To fully support the efficiency chart, we return it here if needed, or the frontend consumes it from the `done` event.
# For simplicity, returning a mock or handling it entirely via stream event.
@router.get("/{id}/efficiency")
async def get_active_learning_efficiency(id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(ActiveLearningRun).where(ActiveLearningRun.id == id)
    result = await db.execute(stmt)
    db_run = result.scalar_one_or_none()
    
    if not db_run:
        return {"success": False, "data": None, "error": {"code": "NOT_FOUND", "message": "Run not found"}}
        
    return {
        "success": True, 
        "data": {
            "efficiency_gain": db_run.efficiency_vs_random,
            "final_max_d33": db_run.final_max_d33,
            "iterations_to_max": db_run.found_max_d33_at_iter
        }, 
        "error": None
    }
