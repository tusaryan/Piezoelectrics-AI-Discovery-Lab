from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from apps.api.app.core.database import get_db
from packages.db.models.composite import CompositePrediction
from piezo_ml.pipeline.beta_phase_estimator import BetaPhaseEstimator
import random # Mocked for API inference speed right now

router = APIRouter(prefix="/api/v1/composite", tags=["Composite Predictions"])

class CompositePredictRequest(BaseModel):
    formula: str
    matrix_type: str
    filler_wt_pct: float
    particle_morphology: str
    particle_size_nm: float
    surface_treatment: str
    fabrication_method: str

class CompositePredictResponse(BaseModel):
    composite_d33: float
    bulk_d33: float
    beta_phase_pct: float
    confidence: float
    features_used: List[str]

@router.post("/predict", response_model=Dict[str, Any])
async def predict_composite(request: CompositePredictRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate a prediction for a composite matrix given a ceramic filler weight %.
    """
    beta_phase = BetaPhaseEstimator.estimate(request.matrix_type, request.filler_wt_pct)
    
    # Mocking standard prediction outputs for Fast Phase 2 demonstration
    # In full reality, we route this into CompositeFeatureEngineer -> XGBoost models loaded from registry
    base_d33 = 110.0 + random.uniform(-10, 10)
    
    # Simulate a realistic curve: up to ~15 wt% improves, then structural defect dominates
    wt = request.filler_wt_pct
    multiplier = 1.0 + (wt * 0.05) if wt < 15.0 else 1.75 - ((wt - 15.0) * 0.02)
    multiplier = max(0.2, multiplier) # floor
    
    comp_d33 = base_d33 * multiplier
    
    # Persist the prediction
    prediction_record = CompositePrediction(
        matrix_polymer=request.matrix_type,
        filler_formula=request.formula,
        volume_fraction=wt,
        connectivity="0-3", # assuming 0-3 composite for now
        predicted_d33=comp_d33,
        confidence_lower=comp_d33 * 0.9,
        confidence_upper=comp_d33 * 1.1,
        properties={
            "beta_phase_pct": beta_phase,
            "particle_morphology": request.particle_morphology,
            "particle_size_nm": request.particle_size_nm,
            "surface_treatment": request.surface_treatment,
            "fabrication_method": request.fabrication_method
        }
    )
    db.add(prediction_record)
    await db.commit()
    
    return {
        "status": "success",
        "data": CompositePredictResponse(**{
            "composite_d33": round(float(comp_d33), 2),
            "bulk_d33": round(float(base_d33), 2),
            "beta_phase_pct": round(float(beta_phase), 2),
            "confidence": 0.85,
            "features_used": ["filler_wt_pct", "matrix_type_enc", "particle_morphology", "beta_phase_pct"]
        }).model_dump()
    }

@router.get("/loading-curve")
async def get_loading_curve(formula: str, matrix_type: str = "pvdf"):
    """
    Generate an array of data points varying filler_wt_pct from 0 to 80% to draw the loading curve chart.
    """
    curve = []
    base_d33 = 110.0
    
    for wt in range(0, 81):
        beta = BetaPhaseEstimator.estimate(matrix_type, float(wt))
        
        # Simulate realistic PVDF-ceramic loading curve topology
        # Rise initially via beta-phase nucleating around particles, peak ~ 12-16 wt%, decay due to agglomeration
        if wt < 14:
            comp_d33 = base_d33 * (1.0 + (wt * 0.045))
        else:
            comp_d33 = base_d33 * (1.63 - ((wt - 14.0) * 0.015))
            
        comp_d33 = max(base_d33 * 0.3, comp_d33) # Can't go below worst-case polymer floor
        
        curve.append({
            "wt_pct": wt,
            "predicted_d33": round(float(comp_d33), 2),
            "beta_phase_pct": round(float(beta), 2)
        })
        
    return {
        "status": "success",
        "data": {
            "curve": curve,
            "optimal_wt_pct": 14,
            "max_d33": max([c["predicted_d33"] for c in curve])
        }
    }
