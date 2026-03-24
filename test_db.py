import asyncio
from apps.api.app.core.database import AsyncSessionLocal
from sqlalchemy import select, func
from packages.db.models.dataset import Material, Dataset

async def main():
    async with AsyncSessionLocal() as db:
        mc = await db.scalar(select(func.count(Material.id)))
        dc = await db.scalar(select(func.count(Dataset.id)))
        print(f"Materials: {mc}, Datasets: {dc}")

asyncio.run(main())
