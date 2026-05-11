"""
Piezo.AI — Application Configuration
======================================
Centralized settings loaded from .env file.
"""

import json
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_VERSION: str = "2.1.0"
    APP_NAME: str = "Piezo.AI"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai"

    # API
    CORS_ORIGINS: str = '["http://localhost:3000"]'
    API_SECRET_KEY: str = "change-me-to-random-256-bit-secret"

    # ML Paths
    MODEL_ARTIFACTS_PATH: str = "./resources/trained-models"
    TRAINING_ARTIFACTS_PATH: str = "./resources/training-artifacts"

    # Feature flags
    ENABLE_COMPOSITE_MODULE: bool = True
    ENABLE_HARDNESS_MODULE: bool = True
    ENABLE_GNN_MODULE: bool = False
    ENABLE_AGENT_MODULE: bool = False

    # LLM Configuration (optional — for AI insights)
    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096
    GEMINI_API_KEY: str = ""  # Alias: auto-used if LLM_API_KEY is empty

    @property
    def effective_llm_api_key(self) -> str:
        """Return the effective API key — LLM_API_KEY takes priority, then GEMINI_API_KEY."""
        return self.LLM_API_KEY or self.GEMINI_API_KEY

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS JSON string into list."""
        try:
            return json.loads(self.CORS_ORIGINS)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Look for .env in project root (two levels up from app/core/)
        extra = "ignore"


# Find the project root .env file
_project_root = Path(__file__).resolve().parents[3]  # apps/api/app/core/ → project root
_env_path = _project_root / ".env"

settings = Settings(_env_file=str(_env_path) if _env_path.exists() else ".env")
