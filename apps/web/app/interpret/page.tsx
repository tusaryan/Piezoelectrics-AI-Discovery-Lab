"use client";

import { useEffect, useState } from "react";
import { SHAPBeeswarmChart } from "@/components/interpret/SHAPBeeswarmChart";
import { SHAPWaterfallChart } from "@/components/interpret/SHAPWaterfallChart";
import { SHAPDependenceChart } from "@/components/interpret/SHAPDependenceChart";
import { PhysicsValidationCard } from "@/components/interpret/PhysicsValidationCard";
import { SymbolicRegressionCard } from "@/components/interpret/SymbolicRegressionCard";
import { GraphEmbeddingsCard } from "@/components/interpret/GraphEmbeddingsCard";
import { BetaWarning } from "@/components/layout/BetaWarning";
import { BrainCircuit, Activity, LineChart } from "lucide-react";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

export default function InterpretPage() {
  const [globalShap, setGlobalShap] = useState<any[]>([]);
  const [localShap, setLocalShap] = useState<any>(null);
  const [physics, setPhysics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadInterpretability = async () => {
      setIsLoading(true);
      try {
        const fetchSafely = async (url: string) => {
          try {
            const res = await fetch(url);
            if (!res.ok) return { status: "error", data: null };
            const text = await res.text();
            try { return JSON.parse(text); }
            catch { return { status: "error", data: null }; }
          } catch { return { status: "error", data: null }; }
        };
        const [globalRes, localRes, physRes] = await Promise.all([
          fetchSafely("/api/v1/interpret/shap/global"),
          fetchSafely("/api/v1/interpret/shap/local/sample_123"),
          fetchSafely("/api/v1/interpret/shap/physics-validation"),
        ]);
        if (globalRes.success === true) setGlobalShap(globalRes.data);
        if (localRes.success === true) setLocalShap(localRes.data);
        if (physRes.success === true) setPhysics(physRes.data);
      } catch (e) {
        console.error("Failed to load interpretability data", e);
      } finally {
        setIsLoading(false);
      }
    };
    loadInterpretability();
  }, []);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-muted-foreground">
        <div className="w-10 h-10 border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-4" />
        <p className="font-medium animate-pulse">Extracting Model Explanations...</p>
      </div>
    );
  }

  const cards: CardDefinition[] = [
    {
      key: "global-shap",
      title: "Global Feature Importance (Beeswarm)",
      icon: <LineChart className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 0, y: 0, w: 8, h: 6, minW: 5, minH: 4 },
      component: (
        <div className="p-2">
          {globalShap.length > 0 ? (
            <SHAPBeeswarmChart data={globalShap} />
          ) : (
            <div className="h-[400px] flex items-center justify-center text-muted-foreground">No data available</div>
          )}
        </div>
      ),
    },
    {
      key: "local-shap",
      title: "Local Prediction Explanation",
      icon: <Activity className="w-4 h-4 text-rose-500" />,
      defaultLayout: { x: 8, y: 0, w: 4, h: 4, minW: 3, minH: 3 },
      component: (
        <div className="p-4">
          {localShap ? (
            <SHAPWaterfallChart data={localShap.contributions} baseline={localShap.baseline} />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">Select material</div>
          )}
        </div>
      ),
    },
    {
      key: "physics",
      title: "Physics Validation",
      icon: <BrainCircuit className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 8, y: 4, w: 4, h: 3, minW: 3, minH: 2 },
      component: (
        <div className="p-4">
          {physics ? (
            <PhysicsValidationCard
              score={physics.alignment_score}
              confirmed={physics.confirmed}
              violated={physics.violated}
            />
          ) : (
            <div className="h-[200px] flex items-center justify-center text-muted-foreground">Loading...</div>
          )}
        </div>
      ),
    },
    {
      key: "dependence",
      title: "Feature Dependence Plot",
      icon: <LineChart className="w-4 h-4 text-sky-500" />,
      defaultLayout: { x: 0, y: 6, w: 6, h: 4, minW: 4, minH: 3 },
      component: (
        <div className="p-4 h-full">
          {globalShap.length > 0 ? (
            <SHAPDependenceChart data={globalShap} />
          ) : (
            <div className="h-[200px] flex items-center justify-center text-muted-foreground">Select material</div>
          )}
        </div>
      ),
    },
    {
      key: "symbolic",
      title: "Symbolic Regression",
      icon: <LineChart className="w-4 h-4 text-amber-500" />,
      defaultLayout: { x: 6, y: 6, w: 6, h: 4, minW: 4, minH: 3 },
      component: (
        <div className="p-4">
          <SymbolicRegressionCard />
        </div>
      ),
    },
    {
      key: "embeddings",
      title: "Graph Embeddings",
      icon: <BrainCircuit className="w-4 h-4 text-violet-500" />,
      defaultLayout: { x: 0, y: 10, w: 6, h: 4, minW: 4, minH: 3 },
      component: (
        <div className="h-[400px]">
          <GraphEmbeddingsCard />
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1400px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-3">
          <BrainCircuit className="w-8 h-8 text-indigo-500" />
          Interpretability Studio
        </h1>
        <p className="text-muted-foreground max-w-3xl">
          Deep dive into model decision boundaries using SHAP explanations and physical validation frameworks.
        </p>
        <div className="mt-3"><BetaWarning /></div>
      </div>
      <DraggableGrid pageKey="interpret" cards={cards} />
    </div>
  );
}
