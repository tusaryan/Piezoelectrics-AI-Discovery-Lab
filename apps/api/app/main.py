"""
Piezo.AI v2.1.0 — FastAPI Backend
==================================
ARCHITECTURAL RULE: This is a DUMB PIPE.
Zero ML logic allowed here. All ML computations, model loading, formula parsing,
feature engineering, and training live exclusively in packages/ml-core/piezo_ml/.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine
from app.core.logging_config import setup_logging

# Set up structured logging BEFORE any other imports
setup_logging()

from app.modules.dataset.router import router as dataset_router
from app.modules.training.router import router as training_router
from app.modules.prediction.router import router as prediction_router
from app.modules.dashboard.router import router as dashboard_router
from app.modules.interpret.router import router as interpret_router
from app.modules.optimization.router import router as optimization_router
from app.modules.settings.router import router as settings_router

import logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup
    print(f"🚀 Piezo.AI API v{settings.APP_VERSION} starting...")
    print(f"📊 Database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'configured'}")
    print(f"📁 Dataset endpoints: /api/v1/datasets")
    print(f"🧠 Training endpoints: /api/v1/training")
    print(f"🔮 Prediction endpoints: /api/v1/predictions")
    print(f"📊 Dashboard endpoints: /api/v1/dashboard")
    print(f"🔍 Interpret endpoints: /api/v1/interpret")
    print(f"🧪 Optimization endpoints: /api/v1/optimization")
    print(f"⚙️  Settings endpoints: /api/v1/settings")

    # ── Startup DB hygiene: purge models with invalid metrics ──
    # Models with NaN/Inf r2_score or rmse crash JSON serialization and
    # break the entire app. This auto-cleanup ensures a clean boot.
    try:
        from app.core.database import async_session_factory
        from sqlalchemy import text
        async with async_session_factory() as db:
            result = await db.execute(text(
                "DELETE FROM trained_models "
                "WHERE r2_score = 'NaN' OR rmse = 'NaN' "
                "   OR r2_score IS NULL OR rmse IS NULL "
                "   OR r2_score = 'Infinity' OR rmse = 'Infinity' "
                "   OR r2_score = '-Infinity' OR rmse = '-Infinity' "
                "RETURNING id, target, algorithm, r2_score, rmse"
            ))
            deleted = result.fetchall()
            await db.commit()
            if deleted:
                print(f"🧹 Startup cleanup: removed {len(deleted)} model(s) with invalid metrics:")
                for row in deleted:
                    print(f"   🗑️  {row[2]}/{row[1]} (id={str(row[0])[:8]}…) — r2={row[3]}, rmse={row[4]}")
            else:
                print("✅ Startup check: all models have valid metrics")
    except Exception as e:
        print(f"⚠️  Startup DB cleanup skipped: {e}")

    yield
    # Shutdown — kill all background processes before closing DB
    from app.modules.interpret.service import kill_all_background_tasks as kill_shap_tasks
    from app.modules.optimization.service import kill_all_background_tasks as kill_optim_tasks
    kill_shap_tasks()
    kill_optim_tasks()
    await engine.dispose()
    print("🛑 Piezo.AI API shutting down...")


app = FastAPI(
    title="Piezo.AI",
    description="AI-driven discovery platform for lead-free piezoelectric materials",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(dataset_router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(training_router, prefix="/api/v1/training", tags=["training"])
app.include_router(prediction_router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(interpret_router, prefix="/api/v1/interpret", tags=["interpret"])
app.include_router(optimization_router, prefix="/api/v1/optimization", tags=["optimization"])
app.include_router(settings_router, prefix="/api/v1/settings", tags=["settings"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": "Piezo.AI",
        "version": settings.APP_VERSION,
    }


@app.get("/api/v1/info")
async def api_info():
    """API information endpoint."""
    return {
        "app": "Piezo.AI",
        "version": settings.APP_VERSION,
        "description": "AI-driven discovery platform for lead-free piezoelectric materials",
        "sections": [
            "dashboard",
            "dataset",
            "train",
            "predict",
            "optimization-lab",
            "interpret",
            "settings",
        ],
    }
