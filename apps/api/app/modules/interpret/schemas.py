from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any, Dict
from uuid import UUID
from datetime import datetime

class SymbolicRegressionStartRequest(BaseModel):
    target: str = "d33"
    iterations: int = 100
    operators: Optional[List[str]] = ["+", "*", "-", "/", "exp", "sin", "cos"]

class SymbolicRegressionEquationSchema(BaseModel):
    complexity: int
    r2: float
    rmse: Optional[float] = None
    equation: str
    latex: str
    
class SymbolicRegressionRunResponse(BaseModel):
    id: UUID
    target: str
    algorithm: str
    best_equation: Optional[str] = None
    best_r2: Optional[float] = None
    runtime_sec: Optional[float] = None
    created_at: datetime
    discovered_equations: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(from_attributes=True)
