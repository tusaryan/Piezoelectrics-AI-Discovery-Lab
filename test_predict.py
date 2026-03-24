import asyncio
from apps.api.app.core.database import async_session_maker
from apps.api.app.modules.prediction.service import PredictionService

async def main():
    async with async_session_maker() as db:
        try:
            res = await PredictionService.predict_single("BaTiO3", None, None, db)
            print(res)
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
