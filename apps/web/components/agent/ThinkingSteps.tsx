"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Check, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThinkingStepsProps {
  steps: string[];
  isActive: boolean;
}

export function ThinkingSteps({ steps, isActive }: ThinkingStepsProps) {
  const [isExpanded, setIsExpanded] = React.useState(true);

  // Auto-collapse when thinking is done
  React.useEffect(() => {
    if (!isActive && steps.length > 0) {
      const timer = setTimeout(() => setIsExpanded(false), 1500);
      return () => clearTimeout(timer);
    }
  }, [isActive, steps.length]);

  if (steps.length === 0 && !isActive) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, height: 0 }}
      animate={{ opacity: 1, y: 0, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "rounded-xl border overflow-hidden transition-colors duration-500",
        isActive
          ? "border-primary/40 bg-primary/5"
          : "border-border bg-card"
      )}
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <div className="relative">
          <Brain size={14} className={cn(isActive && "text-primary")} />
          {isActive && (
            <motion.div
              className="absolute inset-0 rounded-full border border-primary/40"
              animate={{ scale: [1, 1.8, 1], opacity: [0.5, 0, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}
        </div>
        <span className="font-medium">
          {isActive ? "Thinking..." : `Thought process (${steps.length} steps)`}
        </span>

        {/* Shimmer animation during active thinking */}
        {isActive && (
          <motion.div
            className="h-0.5 flex-1 rounded-full bg-gradient-to-r from-transparent via-primary/30 to-transparent"
            animate={{ x: ["-100%", "100%"] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
        )}

        <span className="ml-auto">
          {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </span>
      </button>

      {/* Steps */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-2 space-y-1">
              {steps.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.2 }}
                  className="flex items-start gap-2 text-xs text-muted-foreground"
                >
                  <Check size={10} className="mt-0.5 text-accent flex-shrink-0" />
                  <span>{step}</span>
                </motion.div>
              ))}

              {/* Active thinking indicator */}
              {isActive && (
                <motion.div
                  className="flex items-center gap-2 text-xs text-primary"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="flex gap-0.5"
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    <span className="w-1 h-1 rounded-full bg-primary" />
                    <span className="w-1 h-1 rounded-full bg-primary" />
                    <span className="w-1 h-1 rounded-full bg-primary" />
                  </motion.div>
                  <span>Processing...</span>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
