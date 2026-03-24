from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class SinglePredictRequest(BaseModel):
    formula: str
    sintering_temp: Optional[float] = None
    d33_artifact_id: Optional[str] = None
    tc_artifact_id: Optional[str] = None
    
class SimilarMaterialSchema(BaseModel):
    formula: str
    d33: float
    tc: float
    similarity: float

class UseCaseSchema(BaseModel):
    category: str
    description: str
    icon: str

class PredictResponseData(BaseModel):
    formula: str
    predicted_d33: float
    predicted_tc: float
    d33_lower_95: float
    d33_upper_95: float
    tc_lower_95: float
    tc_upper_95: float
    parsed_features: Dict[str, float]
    similar_materials: List[SimilarMaterialSchema]
    use_case: UseCaseSchema

class SinglePredictResponse(BaseModel):
    data: PredictResponseData
    meta: Dict[str, Any]

class ComparePredictRequest(BaseModel):
    formulas: List[str]

class ComparePredictResponse(BaseModel):
    data: List[PredictResponseData]
    meta: Dict[str, Any]
