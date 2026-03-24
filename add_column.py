import asyncio
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def main():
    db_url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/piezo_v2")
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE model_artifacts ADD COLUMN alias VARCHAR(255);"))
            print("Successfully added alias column.")
        except Exception as e:
            print(f"Error (maybe already exists?): {e}")

asyncio.run(main())
