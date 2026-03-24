"""
FastAPI router for the Piezo.AI Agent module.

Endpoints:
  - POST /api/v1/agent/chat — SSE streaming chat
  - GET  /api/v1/agent/conversations — list conversations
  - GET  /api/v1/agent/conversations/{id} — get conversation
  - DELETE /api/v1/agent/conversations/{id} — delete conversation
  - POST /api/v1/agent/knowledge/index-paper — upload PDF for RAG
  - GET  /api/v1/agent/knowledge/search — search knowledge base
  - GET  /api/v1/agent/knowledge/stats — knowledge base stats
  - GET  /api/v1/agent/provider — current LLM provider info
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.core.database import get_db
from apps.api.app.core.config import settings
from apps.api.app.modules.agent.schemas import (
    ChatRequest,
    KnowledgeSearchRequest,
)
from apps.api.app.modules.agent.service import AgentService

logger = logging.getLogger("piezo.agent.router")

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


# ── Chat ──────────────────────────────────────────────────────────────

@router.post("/chat")
async def agent_chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    SSE streaming chat with the AI Research Assistant.
    
    Returns a stream of server-sent events:
      - type: thinking — agent reasoning step
      - type: text — streamed text content  
      - type: tool_call — agent calling a tool
      - type: tool_result — tool execution result
      - type: done — stream complete
      - type: error — something went wrong
    """
    logger.info("agent.chat.request", extra={"message": request.message[:80]})

    return StreamingResponse(
        AgentService.chat_stream(
            db=db,
            message=request.message,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Conversations ─────────────────────────────────────────────────────

@router.get("/conversations")
async def list_conversations(
    session_id: str = Query(default="default"),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for a session."""
    conversations = await AgentService.get_conversations(db, session_id)
    return {
        "success": True,
        "data": conversations,
        "meta": {"count": len(conversations)},
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single conversation with full message history."""
    conversation = await AgentService.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "success": True,
        "data": conversation,
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    deleted = await AgentService.delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "data": {"deleted": True}}


# ── Knowledge Base ────────────────────────────────────────────────────

@router.post("/knowledge/index-paper")
async def index_paper(file: UploadFile = File(...)):
    """Upload and index a research paper PDF into the RAG knowledge base."""
    logger.info("agent.knowledge.index_paper", extra={"filename": file.filename})
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    try:
        from piezo_ml.rag.knowledge_base import KnowledgeBase
        from piezo_ml.rag.glossary import ensure_glossary_indexed

        kb = KnowledgeBase(settings.chroma_persist_path)
        ensure_glossary_indexed(kb)
        result = kb.index_paper(contents, file.filename)
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        logger.error("agent.knowledge.index_paper.error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/search")
async def search_knowledge(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=20),
    doc_type: Optional[str] = Query(default=None),
):
    """Search the RAG knowledge base (debug endpoint)."""
    try:
        from piezo_ml.rag.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(settings.chroma_persist_path)
        results = kb.search(q, top_k=top_k, doc_type=doc_type)
        return {
            "success": True,
            "data": results,
            "meta": {"query": q, "count": len(results)},
        }
    except Exception as e:
        logger.error("agent.knowledge.search.error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/stats")
async def knowledge_stats():
    """Get knowledge base statistics."""
    try:
        from piezo_ml.rag.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(settings.chroma_persist_path)
        stats = kb.get_stats()
        return {
            "success": True,
            "data": stats,
        }
    except Exception as e:
        logger.error("agent.knowledge.stats.error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Provider Info ─────────────────────────────────────────────────────

@router.get("/provider")
async def get_provider_info():
    """Get current LLM provider configuration."""
    from apps.api.app.modules.agent.llm_provider import get_provider_info as _get_info
    info = _get_info()
    info["voice_enabled"] = settings.enable_voice
    info["voice_provider"] = settings.voice_provider if settings.enable_voice else None
    return {
        "success": True,
        "data": info,
    }
