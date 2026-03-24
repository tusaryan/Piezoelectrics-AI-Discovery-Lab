"use client";

import { useState, useRef } from "react";
import { OptimizationConfig } from "@/components/inverse/OptimizationConfig";
import { ParetoFrontChart } from "@/components/inverse/ParetoFrontChart";
import { ParetoSolutionTable } from "@/components/inverse/ParetoSolutionTable";
import { ConvergenceChart } from "@/components/inverse/ConvergenceChart";
import { Layers, Cuboid, Download, TrendingUp, Table2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

export default function InverseDesignPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [convergenceData, setConvergenceData] = useState<any[]>([]);
  const [solutions, setSolutions] = useState<any[]>([]);
  const [hoveredSolutionId, setHoveredSolutionId] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const startOptimization = async (config: any) => {
    setIsRunning(true);
    setConvergenceData([]);
    setSolutions([]);
    setHoveredSolutionId(null);

    try {
      const res = await fetch("/api/v1/inverse/pareto/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config)
      });
      const data = await res.json();

      if (data.success && data.data.run_id) {
        const sse = new EventSource(`/api/v1/inverse/pareto/${data.data.run_id}/convergence`);
        eventSourceRef.current = sse;

        sse.onmessage = async (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.generation !== undefined) {
              setConvergenceData(prev => [...prev, msg]);
            }
            if (msg.type === "done") {
              const solRes = await fetch(`/api/v1/inverse/pareto/${data.data.run_id}/front`);
              const solData = await solRes.json();
              if (solData.success) setSolutions(solData.data);
              sse.close();
              setIsRunning(false);
            } else if (msg.type === "error") {
              sse.close();
              setIsRunning(false);
            }
          } catch {}
        };

        sse.onerror = () => {
          sse.close();
          setIsRunning(false);
        };
      } else {
        setIsRunning(false);
      }
    } catch {
      setIsRunning(false);
    }
  };

  const handleAddToCart = (id: string) => {
    console.log(`Added solution ${id} to comparison queue`);
  };

  const handleDownloadReport = async () => {
    if (solutions.length === 0) return;
    try {
      const res = await fetch("/api/v1/predict/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: "pareto-run", include_pareto: true })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `pareto_report.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error(e);
    }
  };

  const cards: CardDefinition[] = [
    {
      key: "config",
      title: "Optimization Config",
      icon: <Layers className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 0, y: 0, w: 3, h: 8, minW: 3, minH: 5 },
      component: (
        <div className="p-4 h-full">
          <OptimizationConfig onStart={startOptimization} isRunning={isRunning} />
        </div>
      ),
    },
    {
      key: "pareto-surface",
      title: "Pareto Surface Projection",
      icon: <Cuboid className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 3, y: 0, w: 5, h: 4, minW: 4, minH: 3 },
      component: (
        <div className="p-2 h-full relative">
          {solutions.length > 0 ? (
            <ParetoFrontChart
              data={solutions}
              hoveredId={hoveredSolutionId}
              onHover={setHoveredSolutionId}
              onClick={(id) => console.log('Clicked', id)}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-muted-foreground border-2 border-dashed m-4 rounded-lg">
              Optimization pending.
            </div>
          )}
        </div>
      ),
    },
    {
      key: "convergence",
      title: "Generational Convergence",
      icon: <TrendingUp className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 8, y: 0, w: 4, h: 4, minW: 3, minH: 3 },
      component: (
        <div className="p-4 h-full">
          {convergenceData.length > 0 ? (
            <ConvergenceChart data={convergenceData} />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg">
              Tracking metric history.
            </div>
          )}
        </div>
      ),
    },
    {
      key: "solutions",
      title: "Non-Dominated Candidates",
      icon: <Table2 className="w-4 h-4 text-violet-500" />,
      defaultLayout: { x: 3, y: 4, w: 9, h: 4, minW: 6, minH: 3 },
      component: (
        <div className="h-full flex flex-col overflow-hidden">
          <div className="px-4 py-2 border-b bg-muted/20 flex justify-between items-center">
            <span className="text-xs bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200 px-2.5 py-0.5 rounded-full font-medium">
              {solutions.length} Materials
            </span>
            <Button onClick={handleDownloadReport} variant="outline" size="sm" className="h-7 text-xs gap-1">
              <Download className="w-3 h-3" /> Report
            </Button>
          </div>
          <div className="flex-1 overflow-auto">
            <ParetoSolutionTable
              data={solutions}
              hoveredId={hoveredSolutionId}
              onHover={setHoveredSolutionId}
              onAddToCart={handleAddToCart}
            />
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1600px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-3">
          <Layers className="w-8 h-8 text-indigo-500" />
          Inverse Design Laboratory
        </h1>
        <p className="text-muted-foreground max-w-3xl">
          Discover novel piezoelectric compositions by defining target property bounds and letting the NSGA-II optimizer navigate the ML surrogate model space.
        </p>
      </div>
      <DraggableGrid pageKey="inverse" cards={cards} />
    </div>
  );
}
