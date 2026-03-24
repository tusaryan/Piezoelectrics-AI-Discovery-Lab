from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from packages.db.models.dataset import Dataset, Material
from packages.db.models.training import TrainingJob, ModelArtifact

class SystemService:
    @staticmethod
    async def get_dashboard_stats(db: AsyncSession) -> dict:
        total_materials = await db.scalar(select(func.count(Material.id))) or 0
        total_datasets = await db.scalar(select(func.count(Dataset.id))) or 0
        total_models = await db.scalar(select(func.count(ModelArtifact.id))) or 0
        total_training_jobs = await db.scalar(select(func.count(TrainingJob.id))) or 0
        
        return {
            "total_materials": total_materials,
            "total_datasets": total_datasets,
            "total_models": total_models,
            "total_training_jobs": total_training_jobs
        }
