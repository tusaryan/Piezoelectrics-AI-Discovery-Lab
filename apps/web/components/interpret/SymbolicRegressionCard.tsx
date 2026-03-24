"use client";

import { useState } from "react";
import { Sigma, Play } from "lucide-react";
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

export function SymbolicRegressionCard() {
  const [running, setRunning] = useState(false);
  const [equations, setEquations] = useState<any[]>([]);

  const startPySR = async () => {
    setRunning(true);
    setEquations([]);
    
    try {
      const startRes = await fetch("/api/v1/interpret/symbolic-regression/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: "d33", iterations: 5 })
      });
      const startData = await startRes.json();
      
      const eventSource = new EventSource(`/api/v1/interpret/symbolic-regression/${startData.data.job_id}/stream`);
      
      eventSource.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "done") {
          setEquations(msg.equations || []);
          setRunning(false);
          eventSource.close();
        } else if (msg.type === "error") {
          setRunning(false);
          eventSource.close();
        }
      };
    } catch(e) {
      console.error("PySR Error", e);
      setRunning(false);
    }
  };

  return (
    <div className="rounded-xl border bg-card shadow-sm overflow-hidden h-full flex flex-col">
       <div className="p-6 border-b bg-muted/20 flex justify-between items-center">
          <div>
            <h3 className="font-semibold text-lg flex items-center gap-2">
               <Sigma className="w-5 h-5 text-amber-500" />
               Symbolic Regression (PySR)
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
               Evolves explicit, interpretable algebraic math equations directly fitting the high-dimensional feature landscape.
            </p>
          </div>
          
          <button 
             onClick={startPySR} 
             disabled={running}
             className="px-4 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors disabled:opacity-50"
          >
             {running ? (
                <div className="w-4 h-4 border-2 border-t-amber-500 border-amber-500/20 rounded-full animate-spin"/>
             ) : (
                <Play className="w-4 h-4" />
             )}
             {running ? "Evolving Math..." : "Run Equation Search"}
          </button>
       </div>
       
       <div className="p-6 flex-1 flex flex-col gap-4 bg-[#0a0a0f]">
          {equations.length === 0 && !running && (
             <div className="text-center text-muted-foreground py-10">
                Click "Run Equation Search" to trigger the genetic symbolic regressor.
             </div>
          )}
          
          {running && equations.length === 0 && (
             <div className="text-center text-amber-500/70 py-10 animate-pulse font-mono tracking-widest text-sm">
                [ MUTATING OPERATION TREES ]
             </div>
          )}
          
          {equations.map((eq, i) => (
             <div key={i} className="bg-[#11111b] border border-white/5 rounded-xl p-5 flex flex-col gap-4 shadow-xl">
                <div className="flex justify-between items-center">
                   <span className="text-xs uppercase tracking-wider font-semibold text-amber-500">Complexity: {eq.complexity}</span>
                   <span className="text-xs font-mono bg-emerald-500/10 text-emerald-400 px-2 py-1 rounded">R²: {eq.r2.toFixed(3)}</span>
                </div>
                
                <div className="text-xl overflow-x-auto text-white">
                  {/* Assuming eq.equation is a latex string or python math string. We render standard generic block for now */}
                   <BlockMath math={`d_{33} = ${eq.equation.replace(/\*/g, '\\cdot ').replace(/\//g, '\\div ')}`} />
                </div>
                
                <div className="text-xs font-mono text-muted-foreground line-clamp-2">
                   {eq.description}
                </div>
             </div>
          ))}
       </div>
    </div>
  );
}
