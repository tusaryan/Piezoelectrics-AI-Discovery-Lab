"use client";

import { motion } from "framer-motion";
import { Copy, Terminal } from "lucide-react";
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

interface EquationCardProps {
  id: string;
  complexity: number;
  r2: number;
  rmse?: number;
  equation: string;
  latex: string;
  isSelected?: boolean;
}

export function EquationCard({ id, complexity, r2, rmse, equation, latex, isSelected }: EquationCardProps) {
  
  const handleCopyLatex = () => {
    navigator.clipboard.writeText(latex);
  };

  const handleCopyPython = () => {
    navigator.clipboard.writeText(equation);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`relative rounded-xl border p-5 transition-all duration-300 ease-in-out ${
        isSelected ? "bg-indigo-50/50 border-indigo-400 dark:bg-indigo-950/20 dark:border-indigo-600 shadow-sm" : "bg-card border-border hover:border-indigo-300/50"
      }`}
      id={`eq-${id}`}
    >
        <div className="flex justify-between items-start mb-4">
            <div className="flex gap-3 items-center">
                <span className="inline-flex items-center rounded-full bg-indigo-100 px-2.5 py-0.5 text-xs font-semibold text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
                    Complexity: {complexity}
                </span>
                <span className="text-sm text-muted-foreground font-medium flex items-center gap-1">
                    <span className="text-emerald-500 font-bold">R²:</span> {r2.toFixed(3)}
                </span>
                {rmse !== undefined && (
                   <span className="text-sm text-muted-foreground font-medium flex items-center gap-1">
                      <span className="text-amber-500 font-bold">RMSE:</span> {rmse.toFixed(1)}
                   </span>
                )}
            </div>
            <div className="flex items-center gap-2">
                <button 
                  onClick={handleCopyLatex}
                  className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                  title="Copy LaTeX"
                >
                   <div className="w-4 h-4 text-xs"><BlockMath math="\Sigma" /></div>
                </button>
                <button 
                  onClick={handleCopyPython}
                  className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                  title="Copy Python Code"
                >
                   <Terminal className="w-4 h-4" />
                </button>
            </div>
        </div>

        <div className="flex justify-center items-center overflow-x-auto py-3 px-2 bg-background/50 rounded-lg border border-dashed border-border/60">
            <div className={isSelected ? "text-indigo-700 dark:text-indigo-300" : ""}>
               <BlockMath math={`y = ${latex}`} />
            </div>
        </div>

        <div className="mt-3 text-xs font-mono text-muted-foreground break-all bg-muted/30 p-2 rounded truncate" title={equation}>
            {equation}
        </div>
    </motion.div>
  );
}
