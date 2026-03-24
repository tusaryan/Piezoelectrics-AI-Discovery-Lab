from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.app.core.config import settings
from apps.api.app.core.database import engine
from apps.api.app.core.telemetry import setup_telemetry
from apps.api.app.modules.dataset.router import router as dataset_router
from apps.api.app.modules.training.router import router as training_router
from apps.api.app.modules.prediction.router import router as prediction_router
from apps.api.app.modules.composite.router import router as composite_router
from apps.api.app.modules.hardness.router import router as hardness_router
from apps.api.app.modules.interpret.router import router as interpret_router
from apps.api.app.modules.inverse.router import router as inverse_router
from apps.api.app.modules.active_learning.router import router as active_learning_router
from apps.api.app.modules.system.router import router as system_router

app = FastAPI(
    title="Piezo.AI v2 API",
    description="Backend for AI-Accelerated Discovery of Lead-Free Piezoelectric Composites",
    version="2.0.0"
)

setup_telemetry(app, engine)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
async def health_check():
    return {"success": True, "status": "ok"}

app.include_router(dataset_router)
app.include_router(training_router)
app.include_router(prediction_router)
app.include_router(composite_router)
app.include_router(hardness_router)
app.include_router(interpret_router)
app.include_router(inverse_router)
app.include_router(active_learning_router)
app.include_router(system_router)

# Phase 5: AI Research Assistant (feature-flagged)
if settings.enable_agent_module:
    from apps.api.app.modules.agent.router import router as agent_router
    app.include_router(agent_router)
    
    if settings.enable_voice:
        from apps.api.app.modules.agent.voice_router import router as voice_router
        app.include_router(voice_router)

