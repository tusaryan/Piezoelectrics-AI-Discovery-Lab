from pydantic import BaseModel

class SystemStatsResponse(BaseModel):
    total_materials: int
    total_datasets: int
    total_models: int
    total_training_jobs: int
