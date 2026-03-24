"use client";

import React from "react";
import { motion } from "framer-motion";
import { Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  isStreaming?: boolean;
}

const subscriptMap: Record<string, string> = {
  '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', 
  '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉', 
  '.': '.', 'x': 'ₓ', 'y': 'y', 'z': 'z', '-': '₋', '+': '₊'
};

const toUnicodeSubscript = (str: string) => 
  str.split('').map(c => subscriptMap[c] || c).join('');

const preprocessContent = (text: string) => {
  if (!text) return "";
  return text
    // $_{3}$
    .replace(/\$_{\s*([^{}]+?)\s*}\$/g, (_, m) => toUnicodeSubscript(m))
    // $_3$
    .replace(/\$_\s*([a-zA-Z0-9.\-+]+)\s*\$/g, (_, m) => toUnicodeSubscript(m))
    // ${0.44}$
    .replace(/\$\{\s*([^{}]+?)\s*\}\$/g, (_, m) => toUnicodeSubscript(m))
    // $_3 without trailing $
    .replace(/\$_(\d+)/g, (_, m) => toUnicodeSubscript(m));
};

export function MessageBubble({ role, content, timestamp, isStreaming }: MessageBubbleProps) {
  const [copied, setCopied] = React.useState(false);
  const [showTime, setShowTime] = React.useState(false);
  const isUser = role === "user";

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn("flex w-full group", isUser ? "justify-end" : "justify-start")}
      onMouseEnter={() => setShowTime(true)}
      onMouseLeave={() => setShowTime(false)}
    >
      <div
        className={cn(
          "relative max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-sm"
            : "bg-card border border-border rounded-bl-sm",
          isStreaming && !isUser && "border-primary/30"
        )}
      >
        {/* Content */}
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:my-2 prose-pre:my-2 prose-code:text-xs">
            <ReactMarkdown>{preprocessContent(content)}</ReactMarkdown>
          </div>
        )}

        {/* Streaming cursor */}
        {isStreaming && !isUser && (
          <motion.span
            className="inline-block w-2 h-4 bg-primary/60 rounded-sm ml-0.5"
            animate={{ opacity: [1, 0] }}
            transition={{ duration: 0.8, repeat: Infinity }}
          />
        )}

        {/* Copy button (assistant messages only) */}
        {!isUser && content && !isStreaming && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: showTime ? 1 : 0 }}
            className="absolute -bottom-3 right-2 p-1 rounded bg-secondary hover:bg-secondary/80 text-muted-foreground"
            onClick={handleCopy}
            title="Copy message"
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
          </motion.button>
        )}

        {/* Timestamp on hover */}
        {timestamp && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: showTime ? 0.6 : 0 }}
            className={cn(
              "absolute -bottom-5 text-[10px] text-muted-foreground",
              isUser ? "right-0" : "left-0"
            )}
          >
            {new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
