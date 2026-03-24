"""
Service layer for the Piezo.AI Agent module.

Handles:
  - Streaming chat via LangGraph agent (SSE)
  - Conversation CRUD (create, list, get, update)
  - Knowledge base management
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from apps.api.app.core.config import settings

logger = logging.getLogger("piezo.agent.service")


class AgentService:
    """Service for agent chat and conversation management."""

    # ── Chat Stream ───────────────────────────────────────────────────

    @staticmethod
    async def chat_stream(
        db: AsyncSession,
        message: str,
        conversation_id: Optional[str] = None,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        Invoke the LangGraph agent and yield SSE events.
        
        Event types:
          - thinking: visible reasoning step
          - text: streamed text content
          - tool_call: agent decided to use a tool
          - tool_result: result from tool execution
          - done: stream complete
          - error: something went wrong
        """
        logger.info("agent.chat.start", extra={
            "message": message[:80],
            "conversation_id": conversation_id,
        })

        try:
            from apps.api.app.modules.agent.agent_graph import get_agent_graph

            graph = get_agent_graph()

            # Load or create conversation
            from packages.db.models.prediction import AgentConversation
            
            if conversation_id:
                result = await db.execute(
                    select(AgentConversation).where(AgentConversation.id == conversation_id)
                )
                conversation = result.scalar_one_or_none()
            else:
                conversation = None

            if conversation:
                history = conversation.messages or []
            else:
                conv_id = str(uuid.uuid4())
                conversation = AgentConversation(
                    id=conv_id,
                    session_id=session_id,
                    messages=[],
                    model_used=f"{settings.llm_provider}/{settings.llm_model}",
                )
                db.add(conversation)
                await db.flush()
                history = []
                conversation_id = conv_id

            # Build message history for LangGraph
            langchain_messages = []
            for msg in history:
                if msg.get("role") == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            langchain_messages.append(HumanMessage(content=message))

            # Emit thinking event
            yield _sse_event("thinking", "Analyzing your question...")

            # Stream graph execution
            full_response = ""
            tool_calls_seen = set()

            async for event in graph.astream_events(
                {"messages": langchain_messages, "thinking_steps": []},
                version="v2",
                config={"recursion_limit": 10},
            ):
                kind = event.get("event", "")
                
                # LLM streaming tokens
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, str):
                            full_response += content
                            yield _sse_event("text", content)

                # Tool call start
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    call_id = f"{tool_name}_{id(event)}"
                    if call_id not in tool_calls_seen:
                        tool_calls_seen.add(call_id)
                        yield _sse_event("thinking", f"Using tool: {tool_name}")
                        yield _sse_event("tool_call", {
                            "tool": tool_name,
                            "input": _safe_serialize(tool_input),
                        })

                # Tool result
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    yield _sse_event("tool_result", {
                        "tool": tool_name,
                        "content": _safe_serialize(output),
                    })

            # Update conversation in DB
            history.append({"role": "user", "content": message, "timestamp": _now_iso()})
            history.append({"role": "assistant", "content": full_response, "timestamp": _now_iso()})
            conversation.messages = history
            conversation.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            # Auto-title from first message
            if not conversation.title and message:
                conversation.title = message[:80] + ("..." if len(message) > 80 else "")

            await db.commit()

            yield _sse_event("done", {"conversation_id": str(conversation_id)})
            logger.info("agent.chat.success", extra={
                "conversation_id": conversation_id,
                "response_length": len(full_response),
            })

        except Exception as e:
            logger.error("agent.chat.error", exc_info=True)
            yield _sse_event("error", str(e))
            yield _sse_event("done", {"conversation_id": conversation_id})

    # ── Conversation CRUD ─────────────────────────────────────────────

    @staticmethod
    async def get_conversations(db: AsyncSession, session_id: str = "default") -> list[dict]:
        """List all conversations for a session, sorted by most recent."""
        logger.info("agent.get_conversations", extra={"session_id": session_id})
        from packages.db.models.prediction import AgentConversation

        result = await db.execute(
            select(AgentConversation)
            .where(AgentConversation.session_id == session_id)
            .order_by(desc(AgentConversation.updated_at))
        )
        conversations = result.scalars().all()
        return [
            {
                "id": str(c.id),
                "title": c.title,
                "model_used": c.model_used,
                "message_count": len(c.messages) if c.messages else 0,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in conversations
        ]

    @staticmethod
    async def get_conversation(db: AsyncSession, conversation_id: str) -> Optional[dict]:
        """Get a single conversation with full message history."""
        logger.info("agent.get_conversation", extra={"id": conversation_id})
        from packages.db.models.prediction import AgentConversation

        result = await db.execute(
            select(AgentConversation).where(AgentConversation.id == conversation_id)
        )
        c = result.scalar_one_or_none()
        if not c:
            return None
        return {
            "id": str(c.id),
            "session_id": c.session_id,
            "title": c.title,
            "messages": c.messages or [],
            "model_used": c.model_used,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }

    @staticmethod
    async def delete_conversation(db: AsyncSession, conversation_id: str) -> bool:
        """Delete a conversation."""
        logger.info("agent.delete_conversation", extra={"id": conversation_id})
        from packages.db.models.prediction import AgentConversation

        result = await db.execute(
            select(AgentConversation).where(AgentConversation.id == conversation_id)
        )
        c = result.scalar_one_or_none()
        if not c:
            return False
        await db.delete(c)
        await db.commit()
        return True


# ── Helpers ───────────────────────────────────────────────────────────

def _sse_event(event_type: str, content) -> str:
    """Format an SSE event."""
    data = {"type": event_type, "content": content}
    return f"data: {json.dumps(data)}\n\n"


def _safe_serialize(obj) -> str | dict | list:
    """Safely serialize an object for SSE."""
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    return str(obj)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
