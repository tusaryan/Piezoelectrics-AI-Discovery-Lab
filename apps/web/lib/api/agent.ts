/**
 * API client for the Piezo.AI Agent module.
 * Uses native fetch — no external dependencies.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────

export interface SSEEvent {
  type: "thinking" | "text" | "tool_call" | "tool_result" | "done" | "error";
  content: string | Record<string, unknown>;
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  model_used: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail {
  id: string;
  session_id: string;
  title: string | null;
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    timestamp: string;
  }>;
  model_used: string;
  created_at: string;
  updated_at: string;
}

export interface ProviderInfo {
  provider: string;
  model: string;
  has_api_key: boolean;
  temperature: number;
  max_tokens: number;
  voice_enabled: boolean;
  voice_provider: string | null;
}

// ── Helper ───────────────────────────────────────────────────────────

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const json = await res.json();
  return json.data as T;
}

// ── Chat (SSE Streaming) ─────────────────────────────────────────────

export async function sendChatMessage(
  message: string,
  conversationId?: string | null,
  sessionId: string = "default",
  onEvent?: (event: SSEEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: conversationId || null,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event: SSEEvent = JSON.parse(line.slice(6));
          onEvent?.(event);
        } catch {
          // Skip malformed events
        }
      }
    }
  }
}

// ── Conversations ────────────────────────────────────────────────────

export async function getConversations(
  sessionId: string = "default"
): Promise<ConversationSummary[]> {
  return fetchJson<ConversationSummary[]>(
    `${API_BASE}/api/v1/agent/conversations?session_id=${encodeURIComponent(sessionId)}`
  );
}

export async function getConversation(
  conversationId: string
): Promise<ConversationDetail> {
  return fetchJson<ConversationDetail>(
    `${API_BASE}/api/v1/agent/conversations/${conversationId}`
  );
}

export async function deleteConversation(
  conversationId: string
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/v1/agent/conversations/${conversationId}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
}

// ── Knowledge Base ───────────────────────────────────────────────────

export async function indexPaper(file: File): Promise<{ indexed: number }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/agent/knowledge/index-paper`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Index failed: ${res.status}`);
  const json = await res.json();
  return json.data;
}

export async function searchKnowledge(
  query: string,
  topK: number = 5
): Promise<unknown[]> {
  return fetchJson<unknown[]>(
    `${API_BASE}/api/v1/agent/knowledge/search?q=${encodeURIComponent(query)}&top_k=${topK}`
  );
}

// ── Provider Info ────────────────────────────────────────────────────

export async function getProviderInfo(): Promise<ProviderInfo> {
  return fetchJson<ProviderInfo>(`${API_BASE}/api/v1/agent/provider`);
}
