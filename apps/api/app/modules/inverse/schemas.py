from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

class ParetoOptimizerConfig(BaseModel):
    name: str = "New Inverse Design Run"
    algorithm: str = "NSGA-II"
    objectives: List[str] = ["d33", "tc", "hardness"]
    constraints: Optional[Dict[str, float]] = None
    n_generations: int = 100
    population_size: int = 200

class ParetoRunResponse(BaseModel):
    id: UUID
    name: str
    algorithm: str
    objectives: Dict[str, Any]
    constraints: Optional[Dict[str, Any]]
    n_generations: int
    population_size: int
    status: str
    result_count: Optional[int] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ParetoSolutionResponse(BaseModel):
    id: UUID
    run_id: UUID
    formula: Optional[str] = None
    composition: Dict[str, float]
    predicted_d33: Optional[float]
    predicted_tc: Optional[float]
    predicted_hardness: Optional[float]
    rank: Optional[int]
    crowd_distance: Optional[float]
    use_case: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
