from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.core.database import get_db
from apps.api.app.modules.system.service import SystemService
from apps.api.app.modules.system.schemas import SystemStatsResponse

router = APIRouter(prefix="/api/v1/system", tags=["system"])

@router.get("/stats", response_model=dict)
async def get_system_stats(db: AsyncSession = Depends(get_db)):
    stats = await SystemService.get_dashboard_stats(db)
    return {
        "success": True,
        "data": stats,
        "error": None,
        "meta": {}
    }
