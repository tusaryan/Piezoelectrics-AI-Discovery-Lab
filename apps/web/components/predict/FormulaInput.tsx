"use client";

import { useState } from "react";
import { Search, Sparkles } from "lucide-react";
import { motion } from "framer-motion";

interface FormulaInputProps {
  onPredict: (formula: string) => void;
  isLoading: boolean;
}

export function FormulaInput({ onPredict, isLoading }: FormulaInputProps) {
  const [formula, setFormula] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (formula.trim()) {
      onPredict(formula);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full relative group">
      <div className="absolute inset-0 bg-gradient-to-r from-primary/30 to-indigo-500/30 rounded-2xl blur-xl transition-all duration-500 group-hover:blur-2xl opacity-50" />
      
      <div className="relative flex items-center bg-card rounded-2xl border shadow-lg overflow-hidden">
        <div className="pl-6 w-full py-3">
          <label className="text-xs font-semibold text-primary uppercase tracking-wider block mb-1">
            Chemical Composition
          </label>
          <input
            type="text"
            value={formula}
            onChange={(e) => setFormula(e.target.value)}
            disabled={isLoading}
            placeholder="e.g. BaTiO3 or (K0.5Na0.5)NbO3"
            className="w-full bg-transparent border-none focus:outline-none text-2xl font-mono text-foreground placeholder:text-muted-foreground/30"
            autoComplete="off"
            spellCheck="false"
          />
        </div>
        
        <div className="pr-3 py-3">
          <button
            type="submit"
            disabled={!formula.trim() || isLoading}
            className="h-14 px-6 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-semibold flex flex-col items-center justify-center transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          >
            {isLoading ? (
              <div className="w-5 h-5 rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin" />
            ) : (
              <>
                <Sparkles className="w-5 h-5 mb-0.5" />
                <span className="text-xs font-medium">Predict</span>
              </>
            )}
          </button>
        </div>
      </div>
      
      {/* Helper text highlighting parser capabilities */}
      <div className="mt-4 flex flex-wrap gap-2 justify-center text-xs text-muted-foreground/80">
        <span className="bg-muted px-2 py-1 rounded-md">Supports (parentheses)</span>
        <span className="bg-muted px-2 py-1 rounded-md">Decimals enabled</span>
        <span className="bg-muted px-2 py-1 rounded-md">Auto-fixes unicode sub-scripts</span>
      </div>
    </form>
  );
}
