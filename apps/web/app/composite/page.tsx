"use client";

import { useState, useEffect } from "react";
import { FillerLoadingCurve } from "@/components/composite/FillerLoadingCurve";
import { Layers, Activity, Zap, Info } from "lucide-react";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

export default function CompositePage() {
  const [formData, setFormData] = useState({
    formula: "BaTiO3",
    matrix_type: "pvdf",
    filler_wt_pct: 12,
    particle_morphology: "spherical",
    particle_size_nm: 200,
    surface_treatment: "dopamine",
    fabrication_method: "hot_press",
  });

  interface LoadingCurvePoint { wt_pct: number; predicted_d33: number; beta_phase_pct: number; }
  interface LoadingCurveData { curve: LoadingCurvePoint[]; optimal_wt_pct: number; max_d33: number; }
  interface PredictionData { composite_d33: number; bulk_d33: number; beta_phase_pct: number; confidence: number; features_used: string[]; }

  const [loadingCurveData, setLoadingCurveData] = useState<LoadingCurveData | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    const fetchLoadingCurve = async () => {
      setIsGenerating(true);
      try {
        const res = await fetch(`/api/v1/composite/loading-curve?formula=${formData.formula}&matrix_type=${formData.matrix_type}`);
        const data = await res.json();
        if (data.status === "success") setLoadingCurveData(data.data);
      } catch (e) { console.error("Failed to fetch loading curve", e); }
      setIsGenerating(false);
    };
    fetchLoadingCurve();
  }, [formData.formula, formData.matrix_type]);

  useEffect(() => {
    const getSinglePrediction = async () => {
      try {
        const predictRes = await fetch('/api/v1/composite/predict', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        });
        const text = await predictRes.text();
        let predictData;
        try { predictData = JSON.parse(text); } catch { throw new Error(text || "Backend failed"); }
        if (!predictRes.ok) throw new Error(predictData.detail || "Prediction failed");
        if (predictData.status === "success") setPrediction(predictData.data);
      } catch (e) { console.error("Failed to fetch single prediction", e); }
    };
    if (loadingCurveData) getSinglePrediction();
  }, [formData, loadingCurveData]);

  const handleSetWt = (wt: number) => setFormData(prev => ({ ...prev, filler_wt_pct: wt }));

  const cards: CardDefinition[] = [
    {
      key: "architecture",
      title: "Composite Architecture",
      icon: <Layers className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 0, y: 0, w: 4, h: 6, minW: 3, minH: 4 },
      component: (
        <div className="p-6 space-y-4">
          <div>
            <label className="text-xs font-semibold text-muted-foreground uppercase mb-1.5 block">Ceramic Filler Formula</label>
            <input type="text" value={formData.formula} onChange={(e) => setFormData({...formData, formula: e.target.value})} className="w-full h-10 px-3 rounded-lg border bg-background text-sm font-mono" />
          </div>
          <div>
            <label className="text-xs font-semibold text-muted-foreground uppercase mb-1.5 block">Polymer Matrix</label>
            <select value={formData.matrix_type} onChange={(e) => setFormData({...formData, matrix_type: e.target.value})} className="w-full h-10 px-3 rounded-lg border bg-background text-sm">
              <option value="pvdf">PVDF (Standard)</option>
              <option value="pvdf_trfe">PVDF-TrFE</option>
              <option value="epoxy">Epoxy Resin</option>
              <option value="silicone">Silicone Elastomer</option>
            </select>
          </div>
          <div className="pt-2 border-t mt-4">
            <div className="flex justify-between items-end mb-1.5">
              <label className="text-xs font-semibold text-muted-foreground uppercase">Filler Loading</label>
              <span className="text-indigo-500 font-mono font-bold">{formData.filler_wt_pct} wt%</span>
            </div>
            <input type="range" min="0" max="80" step="1" value={formData.filler_wt_pct} onChange={(e) => handleSetWt(parseInt(e.target.value))} className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-indigo-500" />
          </div>
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div>
              <label className="text-[10px] font-semibold text-muted-foreground uppercase mb-1 block">Morphology</label>
              <select value={formData.particle_morphology} onChange={(e) => setFormData({...formData, particle_morphology: e.target.value})} className="w-full h-8 px-2 rounded-md border bg-background text-xs">
                <option value="spherical">Spherical</option><option value="rod">Rod-like</option><option value="fiber">Fiber</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] font-semibold text-muted-foreground uppercase mb-1 block">Surface Treat.</label>
              <select value={formData.surface_treatment} onChange={(e) => setFormData({...formData, surface_treatment: e.target.value})} className="w-full h-8 px-2 rounded-md border bg-background text-xs">
                <option value="none">None</option><option value="dopamine">Dopamine</option><option value="silane">Silane</option>
              </select>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "loading-curve",
      title: "Filler Loading Curve",
      icon: <Activity className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 4, y: 0, w: 8, h: 4, minW: 5, minH: 3 },
      component: (
        <div className="p-4">
          <FillerLoadingCurve data={loadingCurveData?.curve || []} optimalWt={loadingCurveData?.optimal_wt_pct || 0} currentWt={formData.filler_wt_pct} onSetWt={handleSetWt} isLoading={isGenerating} />
        </div>
      ),
    },
    {
      key: "d33-result",
      title: "Predicted d₃₃",
      icon: <Activity className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 4, y: 4, w: 4, h: 3, minW: 3, minH: 2 },
      component: (
        <div className="p-6">
          {prediction ? (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold tracking-tighter text-indigo-500">{prediction.composite_d33.toFixed(1)}</span>
                <span className="text-sm text-muted-foreground font-mono">pC/N</span>
              </div>
              <div className="mt-4 pt-3 border-t space-y-1 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Bulk baseline:</span><span className="font-mono">{prediction.bulk_d33.toFixed(1)} pC/N</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Retention:</span><span className="font-mono font-semibold text-emerald-500">{((prediction.composite_d33 / prediction.bulk_d33) * 100).toFixed(1)}%</span></div>
              </div>
            </>
          ) : (
            <div className="h-[100px] flex items-center justify-center text-muted-foreground animate-pulse">Awaiting prediction...</div>
          )}
        </div>
      ),
    },
    {
      key: "beta-phase",
      title: "Est. PVDF β-Phase",
      icon: <Zap className="w-4 h-4 text-teal-500" />,
      defaultLayout: { x: 8, y: 4, w: 4, h: 3, minW: 3, minH: 2 },
      component: (
        <div className="p-6">
          {prediction ? (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold tracking-tighter text-teal-500">{prediction.beta_phase_pct.toFixed(1)}</span>
                <span className="text-sm text-muted-foreground font-mono">%</span>
              </div>
              <div className="mt-4 text-xs text-muted-foreground bg-muted/50 p-3 rounded-lg flex items-start gap-2">
                <Info className="w-4 h-4 shrink-0 text-teal-500 mt-0.5" />
                <p>High β-phase indicates strong electroactive nucleation from {formData.filler_wt_pct}wt% filler interactions.</p>
              </div>
            </>
          ) : (
            <div className="h-[100px] flex items-center justify-center text-muted-foreground animate-pulse">Awaiting prediction...</div>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1400px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Composite Foundry</h1>
        <p className="text-muted-foreground">Design piezocomposites by dispersing ceramic fillers into flexible polymer matrices.</p>
      </div>
      <DraggableGrid pageKey="composite" cards={cards} />
    </div>
  );
}
