"""
Optimization Router — REST endpoints for structural analysis and NSGA-II optimization.

DUMB PIPE: validates requests, delegates to OptimizationService.

Key change: NSGA-II runs in a background subprocess with result caching.
POST /optimize → starts background task, returns immediately
GET  /optimize/status/{task_key} → poll for completion
"""

from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.optimization.schemas import (
    ObjectiveConfig,
    OptimizationModelInfo,
    OptimizationRequest,
    OptimizationResultResponse,
    ParetoSolutionResponse,
    PresetsResponse,
    StructuralAnalysisCompareRequest,
    StructuralAnalysisRequest,
    StructuralDescriptorResponse,
    UseCasePreset,
)
from app.modules.optimization.service import OptimizationService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=list[OptimizationModelInfo])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all trained models available for optimization."""
    service = OptimizationService(db)
    models = await service.get_models()
    return [
        OptimizationModelInfo(
            id=str(m.id),
            display_name=m.display_name,
            target=m.target,
            algorithm=m.algorithm,
            r2_score=m.r2_score,
            rmse=m.rmse,
            n_train_samples=m.n_train_samples,
            n_test_samples=m.n_test_samples,
            feature_dim=m.feature_dim,
            is_default=m.is_default,
        )
        for m in models
    ]


@router.post("/structural-analysis", response_model=StructuralDescriptorResponse)
async def structural_analysis(
    req: StructuralAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze crystal structure of a chemical formula."""
    logger.info(f"[Structure] Analyzing formula: {req.formula}")
    try:
        service = OptimizationService(db)
        result = await service.run_structural_analysis(req.formula)
        return StructuralDescriptorResponse(**result)
    except MemoryError:
        logger.error(f"[Structure] MEMORY ERROR analyzing '{req.formula}'")
        raise HTTPException(
            status_code=503,
            detail=f"Server ran out of memory analyzing '{req.formula}'. Try a simpler formula.",
        )
    except Exception as e:
        logger.error(f"[Structure] Error analyzing '{req.formula}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Structural analysis failed for '{req.formula}': {str(e)}",
        )


@router.post(
    "/structural-analysis/compare",
    response_model=list[StructuralDescriptorResponse],
)
async def structural_comparison(
    req: StructuralAnalysisCompareRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compare structural analysis of multiple formulas."""
    logger.info(f"[Structure] Comparing {len(req.formulas)} formulas: {req.formulas}")
    try:
        service = OptimizationService(db)
        results = await service.run_structural_comparison(req.formulas)
        return [StructuralDescriptorResponse(**r) for r in results]
    except MemoryError:
        logger.error(f"[Structure] MEMORY ERROR comparing {len(req.formulas)} formulas")
        raise HTTPException(
            status_code=503,
            detail="Server ran out of memory during structural comparison. Try fewer formulas.",
        )
    except Exception as e:
        logger.error(f"[Structure] Comparison error for {req.formulas}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Structural comparison failed: {str(e)}",
        )


@router.post("/optimize")
async def run_optimization(
    req: OptimizationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start NSGA-II optimization (background) or return cached result.

    Returns immediately with one of:
    - Full OptimizationResultResponse (if cached)
    - {"status": "computing", "task_key": "..."} (if started/running)
    - {"status": "error", "error": "..."} (if failed)
    """
    # --- Pre-flight validation ---
    valid_model_ids = {
        target: mid for target, mid in req.model_ids.items()
        if mid and mid.strip()
    }
    skipped_targets = [
        t for t in req.model_ids if t not in valid_model_ids
    ]

    if skipped_targets:
        logger.info(f"[Optimize] Skipped targets (no model selected): {skipped_targets}")

    if not valid_model_ids:
        logger.warning("[Optimize] No valid model IDs provided — all targets skipped")
        return OptimizationResultResponse(
            error="No models selected for optimization. Select at least one surrogate model.",
        )

    targets_str = ", ".join(f"{t}={mid[:8]}…" for t, mid in valid_model_ids.items())
    logger.info(
        f"[Optimize] Starting NSGA-II: targets=[{targets_str}], "
        f"preset={req.preset}, pop={req.pop_size}, gen={req.n_generations}"
    )

    try:
        service = OptimizationService(db)

        # Convert Pydantic ObjectiveConfig to plain dicts
        objectives = {}
        for target, obj in req.objectives.items():
            if target in valid_model_ids:
                objectives[target] = {
                    "direction": obj.direction,
                    "min": obj.min,
                    "max": obj.max,
                    "weight": obj.weight,
                }

        # Default objectives if none provided
        if not objectives:
            for target in valid_model_ids:
                objectives[target] = {
                    "direction": "maximize",
                    "min": 0,
                    "max": 1000,
                    "weight": 1.0,
                }

        result = await service.start_optimization_background(
            model_ids=valid_model_ids,
            objectives=objectives,
            preset=req.preset,
            pop_size=req.pop_size,
            n_generations=req.n_generations,
            seed=req.seed,
            search_elements=req.search_elements,
        )

        if result.get("status") == "completed" and result.get("result"):
            # Return full response from cache
            data = result["result"]
            logger.info(f"[Optimize] Returning cached result ({len(data.get('solutions', []))} solutions)")
            return OptimizationResultResponse(
                solutions=[ParetoSolutionResponse(**s) for s in data["solutions"]],
                convergence=data["convergence"],
                n_generations_run=data["n_generations_run"],
                n_evaluations=data["n_evaluations"],
                duration_seconds=data["duration_seconds"],
                targets_optimized=data["targets_optimized"],
                preset_used=data["preset_used"],
                error=data.get("error"),
            )

        if result.get("status") == "error":
            return OptimizationResultResponse(error=result.get("error", "Unknown error"))

        # Return computing status
        return result

    except MemoryError:
        logger.error(
            f"[Optimize] MEMORY ERROR with pop_size={req.pop_size}, "
            f"n_generations={req.n_generations}"
        )
        return OptimizationResultResponse(
            error=(
                "Server ran out of memory during optimization. "
                f"Try reducing Population Size (currently {req.pop_size}) or "
                f"Generations (currently {req.n_generations})."
            ),
        )
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"[Optimize] {error_type}: {e}")
        logger.error(traceback.format_exc())
        return OptimizationResultResponse(
            error=f"Optimization failed ({error_type}): {str(e)}",
        )


@router.get("/optimize/status/{task_key}")
async def optimization_status(
    task_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Poll status of a background optimization computation.

    Returns:
    - {"status": "not_started"}
    - {"status": "optimizing", "targets": [...], ...}
    - {"status": "completed", "result": { full optimization data }}
    - {"status": "error", "error": "..."}
    """
    service = OptimizationService(db)
    return service.get_task_status(task_key)


@router.get("/presets", response_model=PresetsResponse)
async def get_presets():
    """Get available use-case preset configurations."""
    from piezo_ml.optimization import USE_CASE_PRESETS

    presets = []
    for key, preset in USE_CASE_PRESETS.items():
        objectives = {}
        for target, obj in preset["objectives"].items():
            objectives[target] = ObjectiveConfig(
                direction=obj["direction"],
                min=obj["min"],
                max=obj["max"],
                weight=obj["weight"],
            )
        presets.append(UseCasePreset(
            key=key,
            label=preset["label"],
            description=preset["description"],
            objectives=objectives,
        ))
    return PresetsResponse(presets=presets)
