from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, conint, confloat
from datetime import datetime

class DataIssueSchema(BaseModel):
    row_idx: int
    column: str
    issue_type: str
    severity: str
    description: str
    auto_fixable: bool
    choices: List[str]

class MaterialSchema(BaseModel):
    id: str
    dataset_id: str
    formula: str
    sintering_temp: Optional[float] = None
    d33: Optional[float] = None
    tc: Optional[float] = None
    is_imputed: bool = False
    is_tc_ai_generated: bool = False

class MaterialUpdate(BaseModel):
    formula: Optional[str] = None
    sintering_temp: Optional[float] = None
    d33: Optional[float] = None
    tc: Optional[float] = None

class MaterialListResponse(BaseModel):
    data: List[MaterialSchema]
    meta: Dict[str, Any]

class DatasetSchema(BaseModel):
    id: str
    name: str
    status: str
    row_count: int
    has_d33: bool
    has_tc: bool
    created_at: datetime

class DatasetListResponse(BaseModel):
    data: List[DatasetSchema]
    meta: Dict[str, Any]

class DatasetDetailSchema(DatasetSchema):
    issues: List[DataIssueSchema] = []

class DatasetDetailResponse(BaseModel):
    data: DatasetDetailSchema
    meta: Dict[str, Any] = {}

class ResolveIssueRequest(BaseModel):
    resolutions: Dict[str, str]

class DatasetIssuesResponse(BaseModel):
    data: List[DataIssueSchema]
    meta: Dict[str, Any]

class StandardError(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class StandardResponse(BaseModel):
    data: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[StandardError] = None
