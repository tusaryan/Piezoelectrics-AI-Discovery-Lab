"""
WebSocket voice proxy for the Piezo.AI Agent.

Supports:
  - OpenAI Realtime API (WebSocket relay)
  - Google Gemini Live API (WebSocket relay)

Gated by ENABLE_VOICE feature flag.
Frontend sends audio chunks via WebSocket, backend proxies to the
chosen voice API and relays transcriptions/audio responses back.
"""
import json
import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.api.app.core.config import settings

logger = logging.getLogger("piezo.agent.voice")

router = APIRouter(prefix="/api/v1/agent", tags=["agent-voice"])


@router.websocket("/voice")
async def voice_websocket(ws: WebSocket):
    """
    WebSocket endpoint for real-time voice interaction.
    
    Protocol:
      Client → Server:
        {"type": "audio", "data": "<base64-encoded-audio-chunk>"}
        {"type": "config", "voice": "alloy", "language": "en"}
        {"type": "stop"}
      
      Server → Client:
        {"type": "transcript", "content": "user said..."}
        {"type": "response_audio", "data": "<base64-encoded-audio>"}
        {"type": "response_text", "content": "PiezoBot says..."}
        {"type": "error", "content": "..."}
        {"type": "status", "content": "listening|processing|speaking"}
    """
    if not settings.enable_voice:
        await ws.close(code=4003, reason="Voice interaction is disabled. Set ENABLE_VOICE=true in .env")
        return

    await ws.accept()
    logger.info("voice.websocket.connected")

    try:
        provider = settings.voice_provider.lower()

        if provider == "openai":
            await _handle_openai_realtime(ws)
        elif provider == "google":
            await _handle_google_live(ws)
        else:
            await ws.send_json({
                "type": "error",
                "content": f"Unknown voice provider: {provider}. Use 'openai' or 'google'.",
            })

    except WebSocketDisconnect:
        logger.info("voice.websocket.disconnected")
    except Exception as e:
        logger.error("voice.websocket.error", exc_info=True)
        try:
            await ws.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def _handle_openai_realtime(ws: WebSocket):
    """Proxy to OpenAI Realtime API via WebSocket."""
    api_key = settings.openai_realtime_api_key or settings.llm_api_key
    if not api_key:
        await ws.send_json({
            "type": "error",
            "content": "OPENAI_REALTIME_API_KEY not set. Configure it in .env to use voice.",
        })
        return

    try:
        import websockets
    except ImportError:
        await ws.send_json({
            "type": "error",
            "content": "websockets package not installed.",
        })
        return

    # OpenAI Realtime API endpoint
    openai_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    await ws.send_json({"type": "status", "content": "connecting"})

    try:
        async with websockets.connect(openai_url, additional_headers=headers) as openai_ws:
            # Send session config
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": (
                        "You are PiezoBot, an expert AI research assistant for piezoelectric materials. "
                        "Help researchers discover lead-free piezoelectric materials. "
                        "Be scientific and precise but approachable. Keep responses concise for voice."
                    ),
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                    },
                },
            }
            await openai_ws.send(json.dumps(session_config))
            await ws.send_json({"type": "status", "content": "listening"})

            # Bidirectional relay
            async def relay_from_client():
                """Forward audio from browser to OpenAI."""
                async for message in ws.iter_json():
                    msg_type = message.get("type", "")
                    if msg_type == "audio":
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": message["data"],
                        }))
                    elif msg_type == "stop":
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.commit",
                        }))

            async def relay_from_openai():
                """Forward responses from OpenAI to browser."""
                async for raw in openai_ws:
                    data = json.loads(raw)
                    event_type = data.get("type", "")

                    if event_type == "response.audio.delta":
                        await ws.send_json({
                            "type": "response_audio",
                            "data": data.get("delta", ""),
                        })
                    elif event_type == "response.audio_transcript.delta":
                        await ws.send_json({
                            "type": "response_text",
                            "content": data.get("delta", ""),
                        })
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        await ws.send_json({
                            "type": "transcript",
                            "content": data.get("transcript", ""),
                        })
                    elif event_type == "response.done":
                        await ws.send_json({
                            "type": "status",
                            "content": "listening",
                        })
                    elif event_type == "error":
                        await ws.send_json({
                            "type": "error",
                            "content": data.get("error", {}).get("message", "Unknown error"),
                        })

            await asyncio.gather(relay_from_client(), relay_from_openai())

    except Exception as e:
        logger.error("voice.openai.error", exc_info=True)
        await ws.send_json({"type": "error", "content": f"OpenAI connection failed: {str(e)}"})


async def _handle_google_live(ws: WebSocket):
    """Proxy to Google Gemini Live API via WebSocket."""
    api_key = settings.google_live_api_key
    if not api_key:
        await ws.send_json({
            "type": "error",
            "content": "GOOGLE_LIVE_API_KEY not set. Configure it in .env to use Gemini voice.",
        })
        return

    try:
        import websockets
    except ImportError:
        await ws.send_json({
            "type": "error",
            "content": "websockets package not installed.",
        })
        return

    # Gemini Live API endpoint
    gemini_url = (
        f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta"
        f".GenerativeService.BidiGenerateContent?key={api_key}"
    )

    await ws.send_json({"type": "status", "content": "connecting"})

    try:
        async with websockets.connect(gemini_url) as gemini_ws:
            # Send setup message
            setup = {
                "setup": {
                    "model": "models/gemini-2.0-flash-exp",
                    "generation_config": {
                        "response_modalities": ["AUDIO", "TEXT"],
                        "speech_config": {
                            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
                        },
                    },
                    "system_instruction": {
                        "parts": [{
                            "text": (
                                "You are PiezoBot, an expert AI research assistant for piezoelectric materials. "
                                "Help researchers discover lead-free piezoelectric materials. "
                                "Be scientific and precise but approachable. Keep responses concise."
                            ),
                        }],
                    },
                },
            }
            await gemini_ws.send(json.dumps(setup))

            # Wait for setup complete
            setup_response = await gemini_ws.recv()
            await ws.send_json({"type": "status", "content": "listening"})

            # Bidirectional relay
            async def relay_from_client():
                async for message in ws.iter_json():
                    msg_type = message.get("type", "")
                    if msg_type == "audio":
                        await gemini_ws.send(json.dumps({
                            "realtime_input": {
                                "media_chunks": [{
                                    "data": message["data"],
                                    "mime_type": "audio/pcm",
                                }],
                            },
                        }))
                    elif msg_type == "text":
                        await gemini_ws.send(json.dumps({
                            "client_content": {
                                "turns": [{"role": "user", "parts": [{"text": message.get("content", "")}]}],
                                "turn_complete": True,
                            },
                        }))

            async def relay_from_gemini():
                async for raw in gemini_ws:
                    data = json.loads(raw)
                    server_content = data.get("serverContent", {})
                    parts = server_content.get("modelTurn", {}).get("parts", [])

                    for part in parts:
                        if "text" in part:
                            await ws.send_json({
                                "type": "response_text",
                                "content": part["text"],
                            })
                        if "inlineData" in part:
                            await ws.send_json({
                                "type": "response_audio",
                                "data": part["inlineData"].get("data", ""),
                                "mime_type": part["inlineData"].get("mimeType", "audio/pcm"),
                            })

                    if server_content.get("turnComplete"):
                        await ws.send_json({
                            "type": "status",
                            "content": "listening",
                        })

            await asyncio.gather(relay_from_client(), relay_from_gemini())

    except Exception as e:
        logger.error("voice.google.error", exc_info=True)
        await ws.send_json({"type": "error", "content": f"Gemini connection failed: {str(e)}"})
