from typing import Optional
from pydantic_settings import BaseSettings

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
env_path = ROOT_DIR / ".env"

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai"
    
    # ML
    model_artifacts_path: str = str(ROOT_DIR / "data/models")
    
    # API
    api_secret_key: str = "dev-secret-key"
    
    # Feature Flags
    enable_composite_module: bool = True
    enable_hardness_module: bool = True
    enable_gnn_module: bool = False
    enable_agent_module: bool = False
    
    # Agent — Model Agnostic LLM
    llm_provider: str = "openai"  # openai|anthropic|google|ollama
    llm_model: str = "gpt-4o"     # model name per provider
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None  # For ollama: http://localhost:11434
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    
    # Agent — RAG
    chroma_persist_path: str = str(ROOT_DIR / "data/chroma")
    
    # Agent — Voice (optional)
    enable_voice: bool = False
    voice_provider: str = "openai"  # openai|google
    openai_realtime_api_key: Optional[str] = None
    google_live_api_key: Optional[str] = None

    class Config:
        env_file = str(env_path)
        extra = "ignore"

    def model_post_init(self, __context) -> None:
        # Resolve relative paths against ROOT_DIR
        if not os.path.isabs(self.model_artifacts_path):
            object.__setattr__(self, 'model_artifacts_path', str(ROOT_DIR / self.model_artifacts_path))
        if not os.path.isabs(self.chroma_persist_path):
            object.__setattr__(self, 'chroma_persist_path', str(ROOT_DIR / self.chroma_persist_path))

        # Fallback: if the paths are not writable (e.g. /data on macOS), use project-local data/
        for attr, subdir in [('model_artifacts_path', 'data/models'), ('chroma_persist_path', 'data/chroma')]:
            path = getattr(self, attr)
            parent = os.path.dirname(path) if os.path.dirname(path) != path else path
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                fallback = str(ROOT_DIR / subdir)
                os.makedirs(fallback, exist_ok=True)
                object.__setattr__(self, attr, fallback)

settings = Settings()
