"use client";

import { useState } from "react";
import { CopyPlus, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PredictionGauges } from "@/components/predict/PredictionGauges";

interface ComparePanelProps {
  onAddCompare: (formula: string) => void;
  isLoading: boolean;
}

export function ComparePanel({ onAddCompare, isLoading }: ComparePanelProps) {
  const [formulas, setFormulas] = useState<string[]>([]);
  const [current, setCurrent] = useState("");

  const add = (e: React.FormEvent) => {
    e.preventDefault();
    if (current.trim() && !formulas.includes(current.trim()) && formulas.length < 5) {
      setFormulas([...formulas, current.trim()]);
      setCurrent("");
    }
  };

  const executeCompare = () => {
    if (formulas.length > 0) {
      // In a real implementation this fires to /api/v1/predict/compare
      // Simulated for mock
      onAddCompare(formulas.join(","));
    }
  };

  return (
    <div className="rounded-2xl border bg-card p-6 mt-8">
      <h3 className="font-semibold text-lg flex items-center gap-2 mb-4">
        <CopyPlus className="w-5 h-5 text-indigo-500" />
        Multi-Material Array Comparison
      </h3>
      
      <form onSubmit={add} className="flex gap-2">
        <input
          type="text"
          value={current}
          onChange={e => setCurrent(e.target.value)}
          placeholder="Add formula to compare queue (e.g. LiNbO3)"
          className="flex-1 h-10 px-3 rounded-lg border bg-background text-sm"
          disabled={formulas.length >= 5}
        />
        <button 
          type="submit"
          disabled={formulas.length >= 5 || !current.trim()}
          className="h-10 px-4 bg-muted hover:bg-muted-foreground/20 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
        >
          Add
        </button>
      </form>

      <div className="flex flex-wrap gap-2 mt-4 min-h-[40px]">
        <AnimatePresence>
          {formulas.map(f => (
            <motion.div 
              key={f}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="flex items-center gap-2 px-3 py-1.5 bg-primary/10 text-primary rounded-full text-sm font-mono font-semibold"
            >
              {f}
              <button 
                onClick={() => setFormulas(formulas.filter(item => item !== f))}
                className="hover:bg-primary/20 rounded-full p-0.5 transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </motion.div>
          ))}
          {formulas.length === 0 && (
             <span className="text-sm text-muted-foreground/50 self-center">Queue empty. Add formulas to run comparison.</span>
          )}
        </AnimatePresence>
      </div>

      <div className="mt-6 flex justify-end pt-4 border-t">
        <button
          onClick={executeCompare}
          disabled={formulas.length < 2 || isLoading}
          className="h-10 px-6 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
        >
          {isLoading ? "Running Simulation..." : "Run Head-to-Head Comparison"}
        </button>
      </div>
    </div>
  );
}
