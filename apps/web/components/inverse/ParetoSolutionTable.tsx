"use client";

import { useMemo } from "react";
import { Plus } from "lucide-react";

interface ParetoSolutionTableProps {
  data: any[];
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onAddToCart?: (id: string) => void;
}

export function ParetoSolutionTable({ data, hoveredId, onHover, onAddToCart }: ParetoSolutionTableProps) {
  
  // Sort data roughly by the primary metric d33 descending
  const sortedData = useMemo(() => {
     return [...data].sort((a, b) => (b.predicted_d33 || 0) - (a.predicted_d33 || 0));
  }, [data]);

  return (
    <div className="w-full h-full overflow-auto rounded-md border">
      <table className="w-full text-sm text-left">
        <thead className="text-xs text-muted-foreground uppercase bg-muted/40 sticky top-0 z-10">
          <tr>
            <th className="px-4 py-3 font-medium">Use Case</th>
            <th className="px-4 py-3 font-medium text-right">d33</th>
            <th className="px-4 py-3 font-medium text-right">Tc (°C)</th>
            <th className="px-4 py-3 font-medium text-right">Hardness</th>
            <th className="px-4 py-3 font-medium text-center">Actions</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.length === 0 ? (
            <tr>
               <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground italic border-b">
                 No solutions available. Generate a Pareto front.
               </td>
            </tr>
          ) : (
            sortedData.map((row) => {
              const baseRowClass = "border-b transition-colors cursor-default";
              const hoverClass = hoveredId === row.id ? "bg-indigo-50 dark:bg-indigo-950/30" : "hover:bg-muted/30";
              return (
                <tr 
                  key={row.id} 
                  className={`${baseRowClass} ${hoverClass}`}
                  onMouseEnter={() => onHover(row.id)}
                  onMouseLeave={() => onHover(null)}
                >
                  <td className="px-4 py-2 font-medium">
                     <span className="inline-flex items-center rounded-full bg-indigo-100 px-2 py-0.5 text-xs text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
                        {row.use_case}
                     </span>
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-indigo-600 dark:text-indigo-400 font-semibold">{row.predicted_d33?.toFixed(1)}</td>
                  <td className="px-4 py-2 text-right font-mono text-emerald-600 dark:text-emerald-400 font-semibold">{row.predicted_tc?.toFixed(1)}</td>
                  <td className="px-4 py-2 text-right font-mono text-amber-600 dark:text-amber-500 font-semibold">{row.predicted_hardness?.toFixed(2)}</td>
                  <td className="px-4 py-2 text-center">
                    <button 
                      onClick={() => onAddToCart && onAddToCart(row.id)}
                      className="inline-flex h-8 items-center justify-center rounded-md border bg-background px-3 text-xs font-medium shadow-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      title="Add to predicted comparison cache"
                    >
                      <Plus className="w-3 h-3 mr-1" /> Compare
                    </button>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
