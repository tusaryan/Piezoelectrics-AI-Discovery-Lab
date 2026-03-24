from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
import uuid
import asyncio
import json
import time

from piezo_ml.evaluation.shap_analyzer import SHAPAnalyzer
from piezo_ml.symbolic_regression.pysr_runner import PySRRunner
from apps.api.app.core.database import get_db, AsyncSessionLocal
from packages.db.models.prediction import SymbolicRegressionRun
from apps.api.app.modules.interpret.schemas import SymbolicRegressionStartRequest, SymbolicRegressionRunResponse

router = APIRouter(prefix="/api/v1/interpret", tags=["Interpretability"])

MOCK_FEATURES = [
    "Tolerance factor", "A-site electropositivity", "B-site polarizability",
    "avg_bond_covalency", "avg_packing_efficiency", "element_Ba", "element_Ti",
    "element_Ca", "element_Zr", "element_Bi", "element_Na", "element_K",
    "element_Nb", "volume_per_atom", "electronegativity_diff"
]

_SR_STREAMS: Dict[str, asyncio.Queue] = {}

async def run_sr_background_job(job_id: str, request: SymbolicRegressionStartRequest):
    runner = PySRRunner(target=request.target)
    queue = _SR_STREAMS.get(job_id)
    start_time = time.time()
    
    async def stream_cb(msg: dict):
        if queue:
            await queue.put(msg)
            
    try:
        # Mock X, y, feature_names for UI development
        equations = await runner.run_discovery_async(
            X=None, y=None, feature_names=MOCK_FEATURES, iterations=request.iterations, cb=stream_cb
        )
        
        # Save to DB
        async with AsyncSessionLocal() as session:
            stmt = select(SymbolicRegressionRun).where(SymbolicRegressionRun.id == UUID(job_id))
            result = await session.execute(stmt)
            db_run = result.scalar_one_or_none()
            
            if db_run:
                db_run.discovered_equations = equations
                if equations:
                    # equations is sorted by complexity ascending generally
                    best_eq = sorted(equations, key=lambda x: x.get("r2", 0), reverse=True)[0]
                    db_run.best_equation = str(best_eq.get("equation"))
                    db_run.best_r2 = float(best_eq.get("r2", 0))
                
                db_run.runtime_sec = time.time() - start_time
                await session.commit()
            
        if queue:
            await queue.put({"type": "done", "equations": equations})
            
    except Exception as e:
        if queue:
            await queue.put({"type": "error", "message": str(e)})
    finally:
        await asyncio.sleep(5)
        if job_id in _SR_STREAMS:
            del _SR_STREAMS[job_id]


@router.post("/symbolic-regression/start")
async def start_symbolic_regression(
    request: SymbolicRegressionStartRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    new_run = SymbolicRegressionRun(
        id=uuid.uuid4(),
        target=request.target,
        algorithm="PySR"
    )
    db.add(new_run)
    await db.commit()
    await db.refresh(new_run)
    
    job_id_str = str(new_run.id)
    _SR_STREAMS[job_id_str] = asyncio.Queue()
    
    background_tasks.add_task(run_sr_background_job, job_id_str, request)
    
    return {"success": True, "data": {"job_id": job_id_str}, "error": None}

@router.get("/symbolic-regression/{id}")
async def get_symbolic_regression(id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(SymbolicRegressionRun).where(SymbolicRegressionRun.id == id)
    result = await db.execute(stmt)
    db_run = result.scalar_one_or_none()
    
    if not db_run:
        return {"success": False, "data": None, "error": {"code": "NOT_FOUND", "message": "Run not found"}}
        
    return {"success": True, "data": SymbolicRegressionRunResponse.model_validate(db_run).model_dump(mode="json"), "error": None}

@router.get("/symbolic-regression/{id}/stream")
async def stream_symbolic_regression(id: str):
    queue = _SR_STREAMS.get(id)
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

@router.get("/shap/global")
async def get_global_shap():
    res = SHAPAnalyzer.global_shap(model=None, X_train=None, feature_names=MOCK_FEATURES)
    return {"success": True, "data": res, "error": None}

@router.get("/shap/local/{material_id}")
async def get_local_shap(material_id: str):
    res = SHAPAnalyzer.local_shap(model=None, X_train=None, x_single=None, feature_names=MOCK_FEATURES)
    return {"success": True, "data": {"baseline": 110.0, "contributions": res}, "error": None}

@router.get("/shap/physics-validation")
async def get_physics_validation():
    global_res = SHAPAnalyzer.global_shap(model=None, X_train=None, feature_names=MOCK_FEATURES)
    report = SHAPAnalyzer.physics_validation(global_res)
    return {"success": True, "data": report, "error": None}

@router.get("/embeddings")
async def get_structural_embeddings(formula: str = "BaTiO3"):
    """
    Returns simulated deep transfer learning structural CHGNet/ALIGNN graph embeddings
    for 3D visualization or similarity clustering.
    """
    import random
    import math
    
    nodes = []
    edges = []
    
    # Generate some mock 3D coordinates for the atoms
    num_atoms = len(formula) * 2  # arbitrary logic
    for i in range(num_atoms):
        nodes.append({
            "id": i,
            "element": random.choice(["Ba", "Ti", "O", "Pb", "Zr", "Sr"]),
            "x": random.uniform(-5, 5),
            "y": random.uniform(-5, 5),
            "z": random.uniform(-5, 5),
            "latent_activation": random.uniform(0, 1) # Neural network feature heat
        })
        
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            x1, y1, z1 = float(nodes[i]["x"]), float(nodes[i]["y"]), float(nodes[i]["z"])
            x2, y2, z2 = float(nodes[j]["x"]), float(nodes[j]["y"]), float(nodes[j]["z"])
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            if dist < 3.0:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": max(0.1, 1.0 - (dist / 3.0))
                })
                
    return {
        "success": True, 
        "data": {
            "formula": formula,
            "model": "CHGNet-v0.3.0",
            "similarity_to_pzt": float(round(float(random.uniform(0.4, 0.98)), 2)),
            "nodes": nodes,
            "edges": edges
        }, 
        "error": None
    }
