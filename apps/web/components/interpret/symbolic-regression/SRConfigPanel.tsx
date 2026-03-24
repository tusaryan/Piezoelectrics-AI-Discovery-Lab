"use client";

import { useState } from "react";
import { Play, Loader2, Server } from "lucide-react";

interface SRConfigPanelProps {
  onStart: (target: string, iterations: number) => void;
  isRunning: boolean;
  progressLogs: any[];
}

export function SRConfigPanel({ onStart, isRunning, progressLogs }: SRConfigPanelProps) {
  const [target, setTarget] = useState("d33");
  const [iterations, setIterations] = useState(100);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isRunning) {
      onStart(target, iterations);
    }
  };

  return (
    <div className="flex flex-col h-full bg-card border rounded-xl shadow-sm overflow-hidden">
      <div className="p-4 border-b bg-muted/20">
        <h3 className="font-semibold text-lg flex items-center gap-2">
           <Server className="w-5 h-5 text-indigo-500" />
           Discovery Config
        </h3>
        <p className="text-sm text-muted-foreground">Configure the PySR backend settings.</p>
      </div>
      
      <div className="p-4 border-b">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Target Property</label>
            <select 
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={isRunning}
              className="w-full flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <option value="d33">Piezoelectric Constant (d33)</option>
              <option value="tc">Curie Temperature (Tc)</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Max Iterations: {iterations}</label>
            <input 
              type="range" 
              min="10" 
              max="500" 
              step="10"
              value={iterations} 
              onChange={(e) => setIterations(parseInt(e.target.value))}
              disabled={isRunning}
              className="w-full accent-indigo-500"
            />
          </div>

          <button 
            type="submit" 
            disabled={isRunning}
            className="w-full h-10 inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-indigo-600 text-white hover:bg-indigo-700 gap-2"
          >
            {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {isRunning ? "Running PySR..." : "Start Discovery"}
          </button>
        </form>
      </div>

      <div className="flex-1 bg-[#0D0E1A] p-4 overflow-y-auto font-mono text-xs text-gray-300">
        <div className="mb-2 text-indigo-400 font-semibold">{'>'} Terminal Output</div>
        {progressLogs.length === 0 && !isRunning && (
            <div className="text-gray-500 italic">Waiting to start job...</div>
        )}
        {progressLogs.map((log, i) => (
          <div key={i} className="mb-1 leading-relaxed">
            <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span>{" "}
            {log.status === "searching" ? (
                <span>Iteration <span className="text-emerald-400">{log.iteration}</span> | Loss: <span className="text-amber-400">{log.best_loss}</span></span>
            ) : log.type === "done" ? (
                <span className="text-emerald-500 font-bold">Process completed successfully.</span>
            ) : log.type === "error" ? (
                 <span className="text-red-500 font-bold">Error: {log.message}</span>
            ) : (
                <span>{JSON.stringify(log)}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
