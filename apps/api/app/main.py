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
from app.modules.dataset.router import router as dataset_router
from app.modules.training.router import router as training_router
from app.modules.prediction.router import router as prediction_router
from app.modules.dashboard.router import router as dashboard_router


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
    yield
    # Shutdown
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
