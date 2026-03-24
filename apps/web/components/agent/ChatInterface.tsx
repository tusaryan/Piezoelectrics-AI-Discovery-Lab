"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Volume2, VolumeX, Loader2, Bot, ArrowDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { sendChatMessage, type SSEEvent } from "@/lib/api/agent";
import { useSoundEffects } from "@/lib/hooks/useSoundEffects";
import { MessageBubble } from "./MessageBubble";
import { ThinkingSteps } from "./ThinkingSteps";
import { ToolCallCard } from "./ToolCallCard";
import { SuggestedQuestions } from "./SuggestedQuestions";
import { VoiceChat } from "./VoiceChat";

// ── Types ────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  thinkingSteps?: string[];
  toolCalls?: Array<{
    tool: string;
    input?: Record<string, unknown>;
    result?: Record<string, unknown> | string | null;
    isRunning: boolean;
  }>;
}

interface ChatInterfaceProps {
  conversationId: string | null;
  onConversationCreated: (id: string) => void;
  initialMessages?: Array<{ role: "user" | "assistant"; content: string; timestamp: string }>;
  voiceEnabled?: boolean;
  pageContext?: string;
  compact?: boolean;
}

// ── Component ────────────────────────────────────────────────────────

export function ChatInterface({
  conversationId,
  onConversationCreated,
  initialMessages = [],
  voiceEnabled = false,
  pageContext,
  compact = false,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(() =>
    initialMessages.map((m, i) => ({
      id: `init-${i}`,
      ...m,
    }))
  );
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [cooldown, setCooldown] = useState(0);

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const autoScrollRef = useRef(true);
  const { playSound, enabled: soundEnabled, toggle: toggleSound } = useSoundEffects();

  useEffect(() => {
    if (cooldown > 0) {
      const timer = setTimeout(() => setCooldown((c) => c - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [cooldown]);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // ── Auto-scroll ──────────────────────────────────────────────────

  const scrollToBottom = useCallback(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, []);

  useEffect(() => {
    if (autoScrollRef.current) {
      scrollToBottom();
    }
  }, [messages, scrollToBottom]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    autoScrollRef.current = nearBottom;
    setShowScrollBtn(!nearBottom);
  }, []);

  // ── Sync initial messages when conversation changes ────────────
  const prevConvIdRef = useRef<string | null>(conversationId);

  useEffect(() => {
    // If the conversation ID just transitioned from null to a string,
    // it means a new conversation was just created and saved mid-flight.
    // We should NOT clear the local messages we just generated!
    if (prevConvIdRef.current === null && conversationId !== null) {
      prevConvIdRef.current = conversationId;
      return;
    }
    prevConvIdRef.current = conversationId;

    setMessages(
      initialMessages.map((m, i) => ({
        id: `init-${i}`,
        ...m,
      }))
    );
    setIsStreaming(false);
    setIsThinking(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  // ── Send Message ─────────────────────────────────────────────────

  const handleSend = useCallback(
    async (messageText?: string) => {
      const text = (messageText || input).trim();
      if (!text || isStreaming) return;

      setInput("");
      playSound("send");

      // Add user message
      const userMsg: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: text,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      // Prepare assistant message placeholder
      const assistantId = `assistant-${Date.now()}`;
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        thinkingSteps: [],
        toolCalls: [],
      };
      setMessages((prev) => [...prev, assistantMsg]);

      setIsStreaming(true);
      setIsThinking(true);

      try {
        await sendChatMessage(text, conversationId, "default", (event: SSEEvent) => {
          switch (event.type) {
            case "thinking":
              playSound("thinking");
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        thinkingSteps: [
                          ...(m.thinkingSteps || []),
                          event.content as string,
                        ],
                      }
                    : m
                )
              );
              break;

            case "text":
              setIsThinking(false);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + (event.content as string) }
                    : m
                )
              );
              break;

            case "tool_call": {
              playSound("toolCall");
              const tc = event.content as Record<string, unknown>;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        toolCalls: [
                          ...(m.toolCalls || []),
                          {
                            tool: tc.tool as string,
                            input: tc.input as Record<string, unknown>,
                            isRunning: true,
                          },
                        ],
                      }
                    : m
                )
              );
              break;
            }

            case "tool_result": {
              const tr = event.content as Record<string, unknown>;
              setMessages((prev) =>
                prev.map((m) => {
                  if (m.id !== assistantId) return m;
                  const tcs = [...(m.toolCalls || [])];
                  const idx = tcs.findIndex(
                    (t) => t.tool === (tr.tool as string) && t.isRunning
                  );
                  if (idx >= 0) {
                    tcs[idx] = { ...tcs[idx], result: tr.content as Record<string, unknown> | string | null, isRunning: false };
                  }
                  return { ...m, toolCalls: tcs };
                })
              );
              break;
            }

            case "done": {
              playSound("receive");
              setIsStreaming(false);
              setIsThinking(false);
              setCooldown(8); // Rate limit to protect API calls
              const done = event.content as Record<string, unknown>;
              if (done.conversation_id && !conversationId) {
                onConversationCreated(done.conversation_id as string);
              }
              break;
            }

            case "error":
              setIsStreaming(false);
              setIsThinking(false);
              setCooldown(3);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content:
                          m.content ||
                          `⚠️ Error: ${event.content || "Something went wrong. Please try again."}`,
                      }
                    : m
                )
              );
              break;
          }
        });
      } catch {
        setIsStreaming(false);
        setIsThinking(false);
        setCooldown(3);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `⚠️ Connection error. Please check if the API is running.` }
              : m
          )
        );
      }
    },
    [input, isStreaming, conversationId, playSound, onConversationCreated]
  );

  // ── Key handling ─────────────────────────────────────────────────

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Auto-resize textarea ─────────────────────────────────────────

  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
    }
  }, [input]);

  // ── Render ───────────────────────────────────────────────────────

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header — hidden in compact mode (FloatingAgent already has one) */}
      {!compact && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/50 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center">
              <Bot size={16} className="text-primary" />
            </div>
            <div>
              <h2 className="text-sm font-medium text-foreground">AI Agent</h2>
              <p className="text-[10px] text-muted-foreground">
                {isStreaming ? "Thinking..." : "AI Research Assistant"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <VoiceChat enabled={voiceEnabled} apiUrl={apiUrl} />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleSound}
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
              title={soundEnabled ? "Mute sounds" : "Enable sounds"}
            >
              {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            </motion.button>
          </div>
        </div>
      )}

      {/* Messages area */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-4 space-y-4 scroll-smooth"
      >
        {!hasMessages ? (
          <SuggestedQuestions onSelect={handleSend} contextHint={pageContext} />
        ) : (
          <>
            {messages.map((msg) => (
              <div key={msg.id} className="space-y-2">
                {/* Thinking steps (before assistant message) */}
                {msg.role === "assistant" &&
                  msg.thinkingSteps &&
                  msg.thinkingSteps.length > 0 && (
                    <div className="pl-2">
                      <ThinkingSteps
                        steps={msg.thinkingSteps}
                        isActive={isThinking && msg.id === messages[messages.length - 1]?.id}
                      />
                    </div>
                  )}

                {/* Tool calls */}
                {msg.role === "assistant" &&
                  msg.toolCalls &&
                  msg.toolCalls.length > 0 && (
                    <div className="pl-2 space-y-1">
                      {msg.toolCalls.map((tc, i) => (
                        <ToolCallCard
                          key={`${tc.tool}-${i}`}
                          toolName={tc.tool}
                          input={tc.input}
                          result={tc.result}
                          isRunning={tc.isRunning}
                        />
                      ))}
                    </div>
                  )}

                {/* Message bubble (skip empty assistant messages still streaming) */}
                {(msg.content || msg.role === "user") && (
                  <MessageBubble
                    role={msg.role}
                    content={msg.content}
                    timestamp={msg.timestamp}
                    isStreaming={
                      isStreaming &&
                      msg.role === "assistant" &&
                      msg.id === messages[messages.length - 1]?.id
                    }
                  />
                )}
              </div>
            ))}

            {/* Typing indicator when waiting for first token */}
            {isStreaming && messages[messages.length - 1]?.content === "" && !isThinking && (
              <div className="flex justify-start">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex gap-1 px-3 py-2 rounded-xl bg-card border border-border"
                >
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-1.5 h-1.5 rounded-full bg-muted-foreground"
                      animate={{ y: [0, -4, 0] }}
                      transition={{
                        duration: 0.6,
                        repeat: Infinity,
                        delay: i * 0.15,
                      }}
                    />
                  ))}
                </motion.div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Scroll to bottom button */}
      <AnimatePresence>
        {showScrollBtn && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={() => {
              autoScrollRef.current = true;
              scrollToBottom();
            }}
            className="absolute bottom-24 right-8 p-2 rounded-full bg-secondary border border-border shadow-lg hover:bg-secondary/80 transition-colors"
          >
            <ArrowDown size={14} />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Input area */}
      <div className="border-t border-border bg-card/50 backdrop-blur-sm p-3">
        <div className="flex items-end gap-2 max-w-4xl mx-auto">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={cooldown > 0 ? `Please wait ${cooldown}s (rate limit)...` : "Ask about piezoelectric materials..."}
            disabled={isStreaming || cooldown > 0}
            rows={1}
            className={cn(
              "flex-1 resize-none rounded-xl border border-border bg-background px-4 py-2.5",
              "text-sm placeholder:text-muted-foreground",
              "focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40",
              "disabled:opacity-50 transition-all"
            )}
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => handleSend()}
            disabled={!input.trim() || isStreaming || cooldown > 0}
            className={cn(
              "p-2.5 rounded-xl transition-colors flex-shrink-0",
              (input.trim() && !isStreaming && cooldown === 0)
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-secondary text-muted-foreground"
            )}
          >
            {isStreaming ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
          </motion.button>
        </div>
        <p className="text-[10px] text-muted-foreground text-center mt-1.5">
          PiezoAgent can make mistakes. Verify predictions with experimental data.
        </p>
      </div>
    </div>
  );
}
