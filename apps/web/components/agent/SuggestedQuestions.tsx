"use client";

import { motion } from "framer-motion";
import { Beaker, Brain, GitCompare, BarChart3 } from "lucide-react";

interface SuggestedQuestionsProps {
  onSelect: (question: string) => void;
  contextHint?: string;
}

const QUESTIONS = [
  {
    icon: Beaker,
    label: "High-performance discovery",
    question: "What formula gives d33 > 400 pC/N with Tc > 250°C?",
    color: "text-blue-400",
  },
  {
    icon: Brain,
    label: "Understand mechanisms",
    question: "Explain why tantalum improves Curie temperature in KNN systems",
    color: "text-purple-400",
  },
  {
    icon: GitCompare,
    label: "Compare materials",
    question: "Compare K0.5Na0.5NbO3 and K0.48Na0.52Nb0.93Sb0.07O3",
    color: "text-emerald-400",
  },
  {
    icon: BarChart3,
    label: "Interpret predictions",
    question: "What does the SHAP analysis reveal about d33 prediction accuracy?",
    color: "text-amber-400",
  },
];

export function SuggestedQuestions({ onSelect, contextHint }: SuggestedQuestionsProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4 py-12">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
          <Brain size={32} className="text-primary" />
        </div>
        <h2 className="text-xl font-semibold text-foreground">AI Research Assistant</h2>
        <p className="text-sm text-muted-foreground mt-1 max-w-md">
          {contextHint || "Ask anything about piezoelectric materials, predict properties, or explore the dataset."}
        </p>
      </motion.div>

      {/* Question cards */}
      <div className="grid grid-cols-1 min-[400px]:grid-cols-2 gap-2 sm:gap-3 max-w-2xl w-full">
        {QUESTIONS.map((q, i) => (
          <motion.button
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08, duration: 0.3 }}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSelect(q.question)}
            title={q.question}
            className="flex items-start gap-3 p-3 sm:p-4 rounded-xl border border-border bg-card hover:bg-secondary/50 hover:border-primary/20 text-left transition-colors group"
          >
            <q.icon size={16} className={`flex-shrink-0 mt-0.5 ${q.color} group-hover:scale-110 transition-transform`} />
            <div className="flex-1 min-w-0">
              <p className="text-[11px] sm:text-xs font-medium text-muted-foreground mb-0.5 sm:mb-1 truncate">{q.label}</p>
              <p className="text-xs sm:text-sm text-foreground leading-snug break-words max-h-10 group-hover:max-h-40 overflow-hidden transition-all duration-500 ease-in-out">{q.question}</p>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
