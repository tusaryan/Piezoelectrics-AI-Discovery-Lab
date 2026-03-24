"use client";

import { useState, useCallback, useRef } from "react";
import { SRConfigPanel } from "@/components/interpret/symbolic-regression/SRConfigPanel";
import { ComplexityAccuracyPlot } from "@/components/interpret/symbolic-regression/ComplexityAccuracyPlot";
import { EquationCard } from "@/components/interpret/symbolic-regression/EquationCard";
import { FunctionSquare, HelpCircle } from "lucide-react";

export default function SymbolicRegressionPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [progressLogs, setProgressLogs] = useState<any[]>([]);
  const [equations, setEquations] = useState<any[]>([]);
  const [selectedEquationId, setSelectedEquationId] = useState<string | undefined>();
  const eventSourceRef = useRef<EventSource | null>(null);

  const startDiscovery = async (target: string, iterations: number) => {
    setIsRunning(true);
    setProgressLogs([]);
    setEquations([]);
    setSelectedEquationId(undefined);

    try {
      const res = await fetch("/api/v1/interpret/symbolic-regression/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target, iterations })
      });
      const data = await res.json();
      
      if (data.success && data.data.job_id) {
        // Start streaming
        const sse = new EventSource(`/api/v1/interpret/symbolic-regression/${data.data.job_id}/stream`);
        eventSourceRef.current = sse;

        sse.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            setProgressLogs(prev => [...prev, msg].slice(-100)); // Keep last 100 logs
            
            if (msg.type === "done") {
              if (msg.equations) setEquations(msg.equations);
              sse.close();
              setIsRunning(false);
            } else if (msg.type === "error") {
              sse.close();
              setIsRunning(false);
            }
          } catch (e) {}
        };

        sse.onerror = () => {
          sse.close();
          setIsRunning(false);
          setProgressLogs(prev => [...prev, { type: "error", message: "SSE Connection lost." }]);
        };
      } else {
        setIsRunning(false);
        setProgressLogs([{ type: "error", message: data.error?.message || "Failed to start." }]);
      }
    } catch (e: any) {
      setIsRunning(false);
      setProgressLogs([{ type: "error", message: e.message }]);
    }
  };

  const handleDotClick = useCallback((payload: any) => {
    setSelectedEquationId(payload.equation);
    // Scroll element into view smoothly
    const el = document.getElementById(`eq-${payload.equation.replace(/\W/g, '')}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  return (
    <div className="p-6 md:p-10 max-w-[1600px] mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-3">
           <FunctionSquare className="w-8 h-8 text-indigo-500" />
           Symbolic Regression
        </h1>
        <p className="text-muted-foreground max-w-3xl">
          Discover explicit mathematical equations mapping material features to target properties using PySR. Trade off complexity and accuracy to extract physical laws.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 h-[calc(100vh-200px)] min-h-[800px]">
        {/* Left Column: Config & Logs */}
        <div className="flex flex-col h-full lg:col-span-1">
          <SRConfigPanel 
             onStart={startDiscovery} 
             isRunning={isRunning} 
             progressLogs={progressLogs} 
          />
        </div>

        {/* Right Column: Results */}
        <div className="flex flex-col gap-6 lg:col-span-3 h-full overflow-hidden">
          
          <div className="rounded-xl border bg-card shadow-sm h-[40%] flex flex-col">
              <div className="p-4 border-b bg-muted/20 flex justify-between items-center">
                 <h3 className="font-semibold px-2">Complexity vs. Accuracy (Pareto Front)</h3>
                 <span title="Identifies optimal trade-offs between equation simplicity and predictive power (R²)"><HelpCircle className="w-4 h-4 text-muted-foreground mr-2 cursor-help" /></span>
              </div>
              <div className="p-4 flex-1">
                 {equations.length > 0 ? (
                    <ComplexityAccuracyPlot 
                      data={equations} 
                      onDotClick={handleDotClick} 
                      selectedEquationId={selectedEquationId} 
                    />
                 ) : (
                    <div className="w-full h-full flex items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg">
                       Start a discovery run to populate the Pareto front.
                    </div>
                 )}
              </div>
          </div>

          <div className="rounded-xl border bg-card shadow-sm h-[60%] flex flex-col">
              <div className="p-4 border-b bg-muted/20">
                 <h3 className="font-semibold px-2">Equation Leaderboard</h3>
              </div>
              <div className="p-6 flex-1 overflow-y-auto space-y-4 bg-muted/10">
                 {equations.length > 0 ? (
                    equations.map((eq, i) => (
                      <EquationCard 
                        key={i}
                        id={eq.equation.replace(/\W/g, '')}
                        complexity={eq.complexity}
                        r2={eq.r2}
                        rmse={eq.rmse}
                        equation={eq.equation}
                        latex={eq.latex}
                        isSelected={selectedEquationId === eq.equation}
                      />
                    ))
                 ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground">
                       <FunctionSquare className="w-12 h-12 mb-4 opacity-20" />
                       <p>Discovered mathematical models will appear here.</p>
                    </div>
                 )}
              </div>
          </div>

        </div>
      </div>
    </div>
  );
}
