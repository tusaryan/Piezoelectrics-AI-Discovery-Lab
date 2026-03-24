"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, MessageSquare, Trash2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "@/lib/api/agent";

interface AgentSidebarProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  isLoading?: boolean;
}

export function AgentSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  isLoading,
}: AgentSidebarProps) {
  const [deleteConfirm, setDeleteConfirm] = React.useState<string | null>(null);

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (deleteConfirm === id) {
      onDelete(id);
      setDeleteConfirm(null);
    } else {
      setDeleteConfirm(id);
      setTimeout(() => setDeleteConfirm(null), 3000);
    }
  };

  return (
    <div className="flex flex-col h-full border-r border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border">
        <h3 className="text-sm font-medium text-foreground">Conversations</h3>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onNew}
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
        >
          <Plus size={12} />
          New
        </motion.button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin text-muted-foreground" />
          </div>
        )}

        <AnimatePresence>
          {conversations.map((conv) => {
            const isActive = conv.id === activeId;
            return (
              <motion.div
                role="button"
                tabIndex={0}
                key={conv.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                onClick={() => onSelect(conv.id)}
                className={cn(
                  "flex items-start gap-2 w-full px-2.5 py-2 rounded-lg text-left group transition-colors cursor-pointer",
                  isActive
                    ? "bg-primary/10 text-foreground"
                    : "hover:bg-secondary text-muted-foreground hover:text-foreground"
                )}
              >
                <MessageSquare size={14} className="mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">
                    {conv.title || "New conversation"}
                  </p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    {conv.message_count} messages · {formatRelative(conv.updated_at)}
                  </p>
                </div>

                {/* Delete button */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: isActive ? 0.6 : 0 }}
                  whileHover={{ opacity: 1 }}
                  className="opacity-0 group-hover:opacity-60 hover:!opacity-100 flex-shrink-0"
                >
                  <button
                    onClick={(e) => handleDelete(conv.id, e)}
                    className={cn(
                      "p-1 rounded transition-colors",
                      deleteConfirm === conv.id
                        ? "text-red-500 bg-red-500/10"
                        : "hover:text-red-400"
                    )}
                    title={deleteConfirm === conv.id ? "Click again to confirm" : "Delete"}
                  >
                    <Trash2 size={11} />
                  </button>
                </motion.div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {!isLoading && conversations.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-8">
            No conversations yet
          </p>
        )}
      </div>
    </div>
  );
}

function formatRelative(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const diff = Date.now() - date.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  } catch {
    return "";
  }
}
