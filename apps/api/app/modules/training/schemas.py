from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class StartTrainingRequest(BaseModel):
    dataset_id: str
    target: str = "d33"
    model_name: str
    mode: str = "expert" # auto, compare, expert
    params: Dict[str, Any] = {}
    use_optuna: bool = False
    optuna_trials: int = 50

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    message: str = "Training job queued"

class StandardResponse(BaseModel):
    data: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[Any] = None
