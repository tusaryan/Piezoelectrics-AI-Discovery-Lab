"use client";

import { useState } from "react";
import { Play, Loader2, Settings2 } from "lucide-react";

interface OptimizationConfigProps {
  onStart: (config: any) => void;
  isRunning: boolean;
}

export function OptimizationConfig({ onStart, isRunning }: OptimizationConfigProps) {
  const [generations, setGenerations] = useState(100);
  const [population, setPopulation] = useState(200);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isRunning) {
      onStart({
        name: "Discovery Run " + new Date().toLocaleTimeString(),
        algorithm: "NSGA-II",
        objectives: ["d33", "tc", "hardness"],
        n_generations: generations,
        population_size: population
      });
    }
  };

  return (
    <div className="flex flex-col h-full bg-card border rounded-xl shadow-sm overflow-hidden">
      <div className="p-4 border-b bg-muted/20">
        <h3 className="font-semibold text-lg flex items-center gap-2">
           <Settings2 className="w-5 h-5 text-indigo-500" />
           NSGA-II Configuration
        </h3>
        <p className="text-sm text-muted-foreground mt-1">Multi-Objective Genetic Algorithm</p>
      </div>
      
      <div className="p-4 flex-1">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-3">
            <h4 className="text-sm font-semibold uppercase text-muted-foreground">Maximize Objectives</h4>
            <div className="flex flex-col gap-2">
               <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked readOnly className="rounded text-indigo-600 focus:ring-indigo-500" /> Piezoelectric Constant (d33)</label>
               <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked readOnly className="rounded text-indigo-600 focus:ring-indigo-500" /> Curie Temperature (Tc)</label>
               <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked readOnly className="rounded text-indigo-600 focus:ring-indigo-500" /> Vickers Hardness</label>
            </div>
          </div>
          
          <div className="space-y-4 pt-4 border-t">
            <div className="space-y-2">
                <label className="text-sm font-medium flex justify-between">
                   Generations <span>{generations}</span>
                </label>
                <input 
                type="range" min="10" max="500" step="10"
                value={generations} 
                onChange={(e) => setGenerations(parseInt(e.target.value))}
                disabled={isRunning}
                className="w-full accent-indigo-500"
                />
            </div>
            <div className="space-y-2">
                <label className="text-sm font-medium flex justify-between">
                   Population Size <span>{population}</span>
                </label>
                <input 
                type="range" min="50" max="1000" step="50"
                value={population} 
                onChange={(e) => setPopulation(parseInt(e.target.value))}
                disabled={isRunning}
                className="w-full accent-indigo-500"
                />
            </div>
          </div>

          <button 
            type="submit" 
            disabled={isRunning}
            className="w-full mt-6 h-10 inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed gap-2"
          >
            {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {isRunning ? "Running Optimization..." : "Start Optimization Run"}
          </button>
        </form>
      </div>
    </div>
  );
}
