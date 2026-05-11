"""
Interpret Router — REST endpoints for SHAP, Physics Validation, and PySR.

DUMB PIPE: validates requests, delegates to InterpretService.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.interpret.schemas import (
    InterpretModelInfo,
    PhysicsValidationRequest,
    PhysicsValidationResponse,
    ShapBeeswarmRequest,
    ShapBeeswarmResponse,
    ShapDependenceRequest,
    ShapDependenceResponse,
    ShapWaterfallRequest,
    ShapWaterfallResponse,
    SymbolicRegressionRequest,
    SymbolicRegressionResponse,
)
from app.modules.interpret.service import InterpretService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=list[InterpretModelInfo])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all trained models available for interpretation."""
    service = InterpretService(db)
    models = await service.get_models()
    return [
        InterpretModelInfo(
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


@router.post("/shap/beeswarm", response_model=ShapBeeswarmResponse)
async def shap_beeswarm(
    req: ShapBeeswarmRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compute SHAP beeswarm (global feature importance)."""
    try:
        service = InterpretService(db)
        result = await service.run_shap_beeswarm(
            model_id=req.model_id,
            max_samples=req.max_samples,
        )
        return ShapBeeswarmResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"SHAP beeswarm error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SHAP analysis failed: {str(e)}")


@router.post("/shap/waterfall", response_model=ShapWaterfallResponse)
async def shap_waterfall(
    req: ShapWaterfallRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compute SHAP waterfall for a single sample."""
    try:
        service = InterpretService(db)
        result = await service.run_shap_waterfall(
            model_id=req.model_id,
            sample_index=req.sample_index,
        )
        return ShapWaterfallResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"SHAP waterfall error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SHAP analysis failed: {str(e)}")


@router.post("/shap/dependence", response_model=ShapDependenceResponse)
async def shap_dependence(
    req: ShapDependenceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compute SHAP dependence for a specific feature."""
    try:
        service = InterpretService(db)
        result = await service.run_shap_dependence(
            model_id=req.model_id,
            feature_name=req.feature_name,
        )
        return ShapDependenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"SHAP dependence error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SHAP analysis failed: {str(e)}")


@router.post("/physics-validation", response_model=PhysicsValidationResponse)
async def physics_validation(
    req: PhysicsValidationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run physics validation on SHAP data."""
    try:
        service = InterpretService(db)
        result = await service.run_physics_validation(model_id=req.model_id)
        return PhysicsValidationResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Physics validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Physics validation failed: {str(e)}",
        )


@router.post("/symbolic-regression", response_model=SymbolicRegressionResponse)
async def symbolic_regression(
    req: SymbolicRegressionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run PySR symbolic regression."""
    try:
        service = InterpretService(db)
        result = await service.run_symbolic_regression(
            model_id=req.model_id,
            max_complexity=req.max_complexity,
            n_iterations=req.n_iterations,
            timeout_seconds=req.timeout_seconds,
        )
        return SymbolicRegressionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Symbolic regression error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Symbolic regression failed: {str(e)}",
        )

@router.post("/install-pysr")
async def install_pysr(db: AsyncSession = Depends(get_db)):
    """Install PySR backend (downloads Julia and packages)."""
    try:
        service = InterpretService(db)
        result = await service.install_pysr_backend()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PySR install error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Installation failed: {str(e)}")
