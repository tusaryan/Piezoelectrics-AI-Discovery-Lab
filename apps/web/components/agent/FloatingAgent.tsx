"use client";

import React, { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, X, Sparkles, Maximize2, Minimize2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { ChatInterface } from "@/components/agent/ChatInterface";
import { AgentSidebar } from "@/components/agent/AgentSidebar";
import {
  getConversations,
  getConversation,
  deleteConversation,
  type ConversationSummary,
} from "@/lib/api/agent";

// ── Page Context Mapping ─────────────────────────────────────────────

const PAGE_CONTEXTS: Record<string, { label: string; hint: string }> = {
  "/": { label: "Dashboard", hint: "You're on the dashboard. I can help summarize recent activity or start a new analysis." },
  "/dataset": { label: "Dataset Management", hint: "I see you're managing datasets. I can help search materials, explain data patterns, or validate formulas." },
  "/train": { label: "Model Training", hint: "You're training models. I can explain model metrics, suggest hyperparameters, or compare training runs." },
  "/predict": { label: "Prediction", hint: "You're on the prediction page. I can predict properties for new formulas, explain results with SHAP, or suggest compositions." },
  "/interpret": { label: "Interpretability", hint: "You're viewing interpretability analysis. I can explain SHAP values, discuss feature importance, or help understand equations." },
  "/active-learning": { label: "Active Learning", hint: "You're viewing active learning. I can explain acquisition strategies, suggest next experiments, or compare strategies." },
};

function getPageContext(pathname: string): { label: string; hint: string } {
  // Match exact or prefix
  for (const [path, ctx] of Object.entries(PAGE_CONTEXTS)) {
    if (pathname === path || (path !== "/" && pathname.startsWith(path))) {
      return ctx;
    }
  }
  return { label: "Piezo.AI", hint: "I'm PiezoAgent — your AI research assistant. Ask me anything about piezoelectric materials." };
}

// ── Floating Agent Widget ────────────────────────────────────────────

export function FloatingAgent() {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [initialMessages, setInitialMessages] = useState<
    Array<{ role: "user" | "assistant"; content: string; timestamp: string }>
  >([]);
  const [isLoadingConvs, setIsLoadingConvs] = useState(false);

  const pathname = usePathname();
  const pageCtx = getPageContext(pathname);
  const voiceEnabled = process.env.NEXT_PUBLIC_ENABLE_VOICE === "true";

  // ── Conversations ──────────────────────────────────────────────

  const loadConversations = useCallback(async () => {
    setIsLoadingConvs(true);
    try {
      const convs = await getConversations();
      setConversations(convs);
    } catch {
      // API might not be running
    } finally {
      setIsLoadingConvs(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) loadConversations();
  }, [isOpen, loadConversations]);

  const handleSelect = useCallback(async (id: string) => {
    setActiveConvId(id);
    try {
      const conv = await getConversation(id);
      setInitialMessages(conv.messages || []);
    } catch {
      setInitialMessages([]);
    }
    setShowSidebar(false);
  }, []);

  const handleNew = useCallback(() => {
    setActiveConvId(null);
    setInitialMessages([]);
    setShowSidebar(false);
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (activeConvId === id) handleNew();
      } catch { /* ignore */ }
    },
    [activeConvId, handleNew]
  );

  const handleConversationCreated = useCallback(
    (newId: string) => {
      setActiveConvId(newId);
      loadConversations();
    },
    [loadConversations]
  );

  return (
    <>
      {/* ── Floating Action Button ────────────────────────────────── */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            key="fab"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setIsOpen(true)}
            className={cn(
              "fixed bottom-6 right-6 z-50",
              "w-14 h-14 rounded-2xl shadow-lg",
              "bg-gradient-to-br from-primary to-primary/80",
              "text-primary-foreground",
              "flex items-center justify-center",
              "hover:shadow-xl hover:shadow-primary/25 transition-shadow",
              "group"
            )}
            title="Open PiezoAgent"
          >
            <Bot size={24} />
            {/* Sparkle hint animation */}
            <motion.div
              className="absolute -top-1 -right-1"
              animate={{ scale: [1, 1.2, 1], opacity: [0.8, 1, 0.8] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Sparkles size={14} className="text-amber-300" />
            </motion.div>
          </motion.button>
        )}
      </AnimatePresence>

      {/* ── Overlay Panel ─────────────────────────────────────────── */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop (only in expanded mode) */}
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-40 bg-black/20 backdrop-blur-[2px]"
                onClick={() => setIsExpanded(false)}
              />
            )}

            {/* Chat panel */}
            <motion.div
              key="panel"
              initial={{ opacity: 0, y: 40, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 40, scale: 0.95 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className={cn(
                "fixed z-50 flex flex-col",
                "bg-background border border-border rounded-2xl shadow-2xl overflow-hidden",
                isExpanded
                  ? "bottom-4 right-4 left-4 top-4 md:left-auto md:top-4 md:w-[700px]"
                  : "bottom-4 right-4 sm:bottom-6 sm:right-6 w-[calc(100vw-2rem)] sm:w-[420px] h-[600px] max-h-[calc(100vh-6rem)] sm:max-h-[calc(100vh-3rem)]"
              )}
            >
              {/* Panel header */}
              <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-card/80 backdrop-blur-md flex-shrink-0">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Bot size={14} className="text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xs font-semibold text-foreground leading-tight">AI Agent</h3>
                    <p className="text-[9px] text-muted-foreground leading-tight">
                      Context: {pageCtx.label}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-0.5">
                  {/* Toggle sidebar */}
                  <button
                    onClick={() => setShowSidebar(!showSidebar)}
                    className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors text-xs"
                    title="Conversations"
                  >
                    ☰
                  </button>
                  {/* Expand/collapse */}
                  <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="p-1.5 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                    title={isExpanded ? "Minimize" : "Expand"}
                  >
                    {isExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  </button>
                  {/* Close */}
                  <button
                    onClick={() => { setIsOpen(false); setIsExpanded(false); }}
                    className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                    title="Close"
                  >
                    <X size={13} />
                  </button>
                </div>
              </div>

              {/* Body */}
              <div className="flex flex-1 min-h-0">
                {/* Sidebar (collapsible) */}
                <AnimatePresence>
                  {showSidebar && (
                    <motion.div
                      initial={{ width: 0, opacity: 0 }}
                      animate={{ width: 220, opacity: 1 }}
                      exit={{ width: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden flex-shrink-0 border-r border-border"
                    >
                      <AgentSidebar
                        conversations={conversations}
                        activeId={activeConvId}
                        onSelect={handleSelect}
                        onNew={handleNew}
                        onDelete={handleDelete}
                        isLoading={isLoadingConvs}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Chat */}
                <div className="flex-1 min-w-0 overflow-hidden">
                  <ChatInterface
                    conversationId={activeConvId}
                    onConversationCreated={handleConversationCreated}
                    initialMessages={initialMessages}
                    voiceEnabled={voiceEnabled}
                    pageContext={pageCtx.hint}
                    compact
                  />
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
