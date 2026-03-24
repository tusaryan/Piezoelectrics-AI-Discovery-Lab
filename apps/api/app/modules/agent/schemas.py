"""
Pydantic v2 schemas for the Piezo.AI Agent module.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /api/v1/agent/chat"""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID to continue")
    session_id: str = Field(default="default", description="Session identifier")


class IndexPaperRequest(BaseModel):
    """Metadata for paper indexing (file sent as multipart)"""
    filename: Optional[str] = None


class KnowledgeSearchRequest(BaseModel):
    """Request for knowledge base search"""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: Optional[str] = Field(None, description="Filter: paper|material|glossary")


# ── Response Schemas ──────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    """Summary view of a conversation for the sidebar list"""
    id: str
    title: Optional[str] = None
    model_used: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetail(BaseModel):
    """Full conversation with all messages"""
    id: str
    session_id: str
    title: Optional[str] = None
    messages: list[dict]
    model_used: str
    created_at: datetime
    updated_at: datetime


class KnowledgeStats(BaseModel):
    """Knowledge base statistics"""
    total_documents: int
    by_type: dict[str, int]


class AgentProviderInfo(BaseModel):
    """Current LLM provider configuration"""
    provider: str
    model: str
    has_api_key: bool
    temperature: float
    max_tokens: int
    voice_enabled: bool = False
    voice_provider: Optional[str] = None
