"use client";

import { useState, useRef, useEffect } from "react";
import { FlaskConical, Terminal, Activity, Focus, Target } from "lucide-react";
import { ALConfigPanel } from "@/components/active-learning/ALConfigPanel";
import { ALEfficiencyChart } from "@/components/active-learning/ALEfficiencyChart";
import { InteractiveParetoFront } from "@/components/active-learning/InteractiveParetoFront";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

export default function ActiveLearningPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<{ step: number; msg: string; type: string }[]>([]);
  const [progress, setProgress] = useState(0);
  const [alResult, setAlResult] = useState<any>(null);
  const [paretoPoints, setParetoPoints] = useState<any[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/v1/active-learning/pareto")
      .then(r => r.json())
      .then(d => { if (d.success) setParetoPoints(d.data); })
      .catch(e => console.error("Pareto fetch error:", e));
  }, []);

  useEffect(() => {
    if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const startSimulation = async (config: any) => {
    setIsRunning(true);
    setLogs([]);
    setProgress(0);
    setAlResult(null);
    setLogs(prev => [...prev, { step: 0, msg: `Initializing Active Learning with ${config.strategy} strategy...`, type: "info" }]);
    try {
      const startRes = await fetch("/api/v1/active-learning/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config)
      });
      const data = await startRes.json();
      if (!data.success) throw new Error("Failed to start job");
      const jobId = data.data.job_id;
      setLogs(prev => [...prev, { step: 0, msg: `Job ${jobId} registered. Connecting...`, type: "success" }]);
      const eventSource = new EventSource(`/api/v1/active-learning/${jobId}/stream`);
      eventSource.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "progress") {
          setProgress((msg.step / msg.total) * 100);
          setLogs(prev => [...prev, { step: msg.step, msg: msg.message, type: "info" }]);
        } else if (msg.type === "done") {
          setAlResult(msg.result);
          setLogs(prev => [...prev, { step: config.budget, msg: "Simulation complete.", type: "success" }]);
          setIsRunning(false);
          eventSource.close();
        } else if (msg.type === "error") {
          setLogs(prev => [...prev, { step: -1, msg: `ERROR: ${msg.message}`, type: "error" }]);
          setIsRunning(false);
          eventSource.close();
        }
      };
      eventSource.onerror = () => {
        setLogs(prev => [...prev, { step: -1, msg: "Lost connection.", type: "error" }]);
        setIsRunning(false);
        eventSource.close();
      };
    } catch (e: any) {
      setIsRunning(false);
      setLogs(prev => [...prev, { step: -1, msg: `Failed: ${e.message}`, type: "error" }]);
    }
  };

  const cards: CardDefinition[] = [
    {
      key: "config",
      title: "Simulation Config",
      icon: <FlaskConical className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 0, y: 0, w: 3, h: 4, minW: 3, minH: 3 },
      component: (
        <div className="p-4">
          <ALConfigPanel onStart={startSimulation} isRunning={isRunning} />
        </div>
      ),
    },
    {
      key: "terminal",
      title: "Lab Execution Feed",
      icon: <Terminal className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 0, y: 4, w: 3, h: 4, minW: 3, minH: 3 },
      component: (
        <div className="bg-[#0D0E1A] flex flex-col h-full overflow-hidden rounded-lg">
          {isRunning && (
            <div className="w-full bg-white/5 h-1">
              <div className="bg-indigo-500 h-full transition-all duration-300" style={{ width: `${progress}%` }} />
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-2">
            {logs.length === 0 && !isRunning && (
              <p className="text-white/30 italic">Ready. Awaiting simulation kickoff...</p>
            )}
            {logs.map((log, i) => (
              <div key={i} className={`flex gap-3 leading-relaxed ${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-emerald-400' : 'text-gray-300'}`}>
                <span className="opacity-50 select-none">[{new Date().toLocaleTimeString([], {hour12:false})}]</span>
                <span className="break-all">{log.msg}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      ),
    },
    {
      key: "stats",
      title: "Discovery Stats",
      icon: <Target className="w-4 h-4 text-amber-500" />,
      defaultLayout: { x: 3, y: 0, w: 9, h: 2, minW: 6, minH: 2 },
      component: (
        <div className="p-4 grid grid-cols-3 gap-4">
          <div className="bg-muted/30 rounded-xl p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-2"><Target className="w-4 h-4 text-indigo-500" /><span className="text-xs font-medium">Best d33</span></div>
            <div className="text-2xl font-bold font-mono">{alResult ? alResult.final_max_d33 : "---"} <span className="text-xs text-muted-foreground">pC/N</span></div>
          </div>
          <div className="bg-muted/30 rounded-xl p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-2"><Activity className="w-4 h-4 text-emerald-500" /><span className="text-xs font-medium">Efficiency</span></div>
            <div className="text-2xl font-bold font-mono text-emerald-500">{alResult ? `+${alResult.efficiency_gain}%` : "---"}</div>
          </div>
          <div className="bg-muted/30 rounded-xl p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-2"><Focus className="w-4 h-4 text-violet-500" /><span className="text-xs font-medium">Iters to Max</span></div>
            <div className="text-2xl font-bold font-mono">{alResult ? alResult.iterations_to_max.strategy : "---"} <span className="text-xs text-muted-foreground">vs {alResult ? alResult.iterations_to_max.baseline : "---"}</span></div>
          </div>
        </div>
      ),
    },
    {
      key: "learning-curve",
      title: "Learning Curve (Strategy vs Random)",
      icon: <Activity className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 3, y: 2, w: 9, h: 3, minW: 5, minH: 3 },
      component: (
        <div className="p-6">
          <ALEfficiencyChart strategyData={alResult?.strategy_curve || []} baselineData={alResult?.baseline_curve || []} efficiencyGain={alResult?.efficiency_gain || 0} />
        </div>
      ),
    },
    {
      key: "pareto",
      title: "Pareto Optimization Front",
      icon: <Focus className="w-4 h-4 text-violet-500" />,
      defaultLayout: { x: 3, y: 5, w: 9, h: 5, minW: 5, minH: 4 },
      component: (
        <div className="bg-[#000000] p-4">
          {paretoPoints.length > 0 ? (
            <InteractiveParetoFront points={paretoPoints} />
          ) : (
            <div className="h-[400px] flex items-center justify-center text-muted-foreground animate-pulse">Generating 3D Manifold...</div>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1600px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-3">
          <FlaskConical className="w-8 h-8 text-indigo-500" />
          Active Learning Simulator
        </h1>
        <p className="text-muted-foreground max-w-3xl">
          Simulate material discovery loops. The AI selects candidates, maximizing d33 with fewest experiments.
        </p>
      </div>
      <DraggableGrid pageKey="active-learning" cards={cards} />
    </div>
  );
}
