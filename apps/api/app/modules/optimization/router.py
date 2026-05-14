"""
Optimization Router — REST endpoints for structural analysis and NSGA-II optimization.

DUMB PIPE: validates requests, delegates to OptimizationService.
"""

from __future__ import annotations

import logging

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
    try:
        service = OptimizationService(db)
        result = await service.run_structural_analysis(req.formula)
        return StructuralDescriptorResponse(**result)
    except Exception as e:
        logger.error(f"Structural analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Structural analysis failed: {str(e)}",
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
    try:
        service = OptimizationService(db)
        results = await service.run_structural_comparison(req.formulas)
        return [StructuralDescriptorResponse(**r) for r in results]
    except Exception as e:
        logger.error(f"Structural comparison error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Structural comparison failed: {str(e)}",
        )


@router.post("/optimize", response_model=OptimizationResultResponse)
async def run_optimization(
    req: OptimizationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run NSGA-II multi-objective optimization."""
    try:
        service = OptimizationService(db)

        # Convert Pydantic ObjectiveConfig to plain dicts
        objectives = {
            target: {
                "direction": obj.direction,
                "min": obj.min,
                "max": obj.max,
                "weight": obj.weight,
            }
            for target, obj in req.objectives.items()
        }

        result = await service.run_optimization(
            model_ids=req.model_ids,
            objectives=objectives,
            preset=req.preset,
            pop_size=req.pop_size,
            n_generations=req.n_generations,
            seed=req.seed,
            search_elements=req.search_elements,
        )

        return OptimizationResultResponse(
            solutions=[ParetoSolutionResponse(**s) for s in result["solutions"]],
            convergence=result["convergence"],
            n_generations_run=result["n_generations_run"],
            n_evaluations=result["n_evaluations"],
            duration_seconds=result["duration_seconds"],
            targets_optimized=result["targets_optimized"],
            preset_used=result["preset_used"],
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}",
        )


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
