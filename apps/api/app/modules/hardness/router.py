from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional
from piezo_ml.models.use_case_mapper import UseCaseMapper
import random # Mocked for API inference layout

router = APIRouter(prefix="/api/v1/hardness", tags=["Hardness Predictions"])

class HardnessRequest(BaseModel):
    formula: str
    predicted_d33: float
    predicted_tc: float
    matrix_type: Optional[str] = None

class HardnessResponse(BaseModel):
    vickers_hv: float
    mohs: float
    use_case: Dict[str, Any]

@router.post("/predict", response_model=Dict[str, Any])
async def predict_hardness(request: HardnessRequest):
    """
    Generate Vickers and Mohs hardness predictions, then classify the commercial use case.
    """
    # Logical Hardness Calculation based on Piezoelectric properties (Phase 6 principles)
    if request.matrix_type and request.matrix_type.lower() in ["pvdf", "pvdf_trfe"]:
        base_vickers = 15.0 + (request.predicted_tc * 0.05)
    elif request.matrix_type and request.matrix_type.lower() == "epoxy":
        base_vickers = 30.0 + (request.predicted_tc * 0.03)
    elif request.matrix_type and request.matrix_type.lower() == "silicone":
        base_vickers = 5.0
    else:
        # Bulk Ceramic Logic:
        # Higher Tc usually indicates a stiffer lattice -> higher hardness
        # Higher d33 (soft piezoelectrics) indicates easier domain wall motion -> lower hardness
        tc_factor = request.predicted_tc * 0.7
        d33_penalty = request.predicted_d33 * 0.4
        
        # Baseline ceramic assumption
        calculated = 350.0 + tc_factor - d33_penalty
        
        # Explicit structure bonuses mapped from formula space
        formula_upper = request.formula.upper()
        if "LI" in formula_upper and "NB" in formula_upper:
            calculated += 250.0 # LiNbO3 is extremely hard
        elif "BA" in formula_upper and "TI" in formula_upper:
            calculated += 50.0 # BaTiO3 is moderately hard
            
        base_vickers = max(100.0, min(1200.0, calculated))
        
    base_mohs = (base_vickers / 150) + 1.5
    base_mohs = max(1.0, min(10.0, base_mohs))
    
    # Generate business classification
    if request.matrix_type:
        mapping = {
            "use_case": "Flexible/Wearable Electronics",
            "description": "Ideal for conformal biometric sensors and soft robotics due to its extreme flexibility and low polymer modulus.",
            "recommended_applications": ["Wearable Health Monitors", "Soft Robotics", "Flexible Tactile Sensors"],
            "icon": "activity",
            "confidence": 0.95
        }
    else:
        mapping = UseCaseMapper.classify(
            d33=request.predicted_d33,
            tc=request.predicted_tc,
            vickers_hardness=base_vickers
        )
    
    return {
        "status": "success",
        "data": HardnessResponse(**{
            "vickers_hv": round(float(base_vickers), 1),
            "mohs": round(float(base_mohs), 1),
            "use_case": mapping
        }).model_dump()
    }
