"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wrench, ChevronDown, ChevronUp, Loader2, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ToolCallCardProps {
  toolName: string;
  input?: Record<string, unknown>;
  result?: Record<string, unknown> | string | null;
  isRunning?: boolean;
}

const TOOL_LABELS: Record<string, string> = {
  predict_material: "🔮 Predict Material",
  search_dataset: "🔍 Search Dataset",
  get_shap_explanation: "📊 SHAP Explanation",
  suggest_compositions: "💡 Suggest Compositions",
  retrieve_from_literature: "📚 Literature Search",
  compare_compositions: "⚖️ Compare Materials",
  generate_pdf_report: "📄 Generate Report",
};

export function ToolCallCard({ toolName, input, result, isRunning }: ToolCallCardProps) {
  const [showInput, setShowInput] = React.useState(false);
  const [showResult, setShowResult] = React.useState(false);

  const label = TOOL_LABELS[toolName] || `🔧 ${toolName}`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.25 }}
      className={cn(
        "rounded-lg border text-xs overflow-hidden my-2",
        isRunning
          ? "border-primary/30 bg-primary/5"
          : "border-border bg-card"
      )}
    >
      {/* Shimmer border when running */}
      {isRunning && (
        <motion.div
          className="h-[2px] bg-gradient-to-r from-transparent via-primary/50 to-transparent"
          animate={{ x: ["-100%", "100%"] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
      )}

      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2">
        {isRunning ? (
          <Loader2 size={12} className="animate-spin text-primary" />
        ) : (
          <CheckCircle2 size={12} className="text-accent" />
        )}
        <span className="font-medium text-foreground">{label}</span>
        <span className={cn(
          "ml-auto px-1.5 py-0.5 rounded text-[10px] font-medium",
          isRunning
            ? "bg-primary/10 text-primary"
            : "bg-accent/10 text-accent"
        )}>
          {isRunning ? "Running" : "Done"}
        </span>
      </div>

      {/* Input section */}
      {input && (
        <div className="border-t border-border/50">
          <button
            onClick={() => setShowInput(!showInput)}
            className="flex items-center gap-1.5 w-full px-3 py-1.5 text-muted-foreground hover:text-foreground transition-colors"
          >
            <Wrench size={10} />
            <span>Input</span>
            {showInput ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>
          <AnimatePresence>
            {showInput && (
              <motion.pre
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="px-3 pb-2 text-[11px] text-muted-foreground overflow-x-auto font-mono"
              >
                {JSON.stringify(input, null, 2)}
              </motion.pre>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Result section */}
      {result && !isRunning && (
        <div className="border-t border-border/50">
          <button
            onClick={() => setShowResult(!showResult)}
            className="flex items-center gap-1.5 w-full px-3 py-1.5 text-muted-foreground hover:text-foreground transition-colors"
          >
            <CheckCircle2 size={10} />
            <span>Result</span>
            {showResult ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>
          <AnimatePresence>
            {showResult && (
              <motion.pre
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="px-3 pb-2 text-[11px] text-muted-foreground overflow-x-auto font-mono max-h-48 overflow-y-auto"
              >
                {typeof result === "string" ? result : JSON.stringify(result as Record<string, unknown>, null, 2)}
              </motion.pre>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}
