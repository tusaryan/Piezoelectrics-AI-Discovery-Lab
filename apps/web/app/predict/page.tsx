"use client";

import { useEffect, useState } from "react";
import { FormulaInput } from "@/components/predict/FormulaInput";
import { PredictionGauges } from "@/components/predict/PredictionGauges";
import { VickersGauge } from "@/components/hardness/VickersGauge";
import { MohsScale } from "@/components/hardness/MohsScale";
import { ComparePanel } from "@/components/predict/ComparePanel";
import { ModelConsensus } from "@/components/predict/ModelConsensus";
import { BatchPredictPanel } from "@/components/predict/BatchPredictPanel";
import { motion, AnimatePresence } from "framer-motion";
import { Droplet, Cpu, Zap, Download, Activity, Shield, ChevronDown, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

export default function PredictPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [compareResults, setCompareResults] = useState<Record<string, unknown>[] | null>(null);
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [selectedD33, setSelectedD33] = useState<string>("");
  const [selectedTc, setSelectedTc] = useState<string>("");

  useEffect(() => {
    // Fetch available models
    fetch("/api/v1/training/models")
      .then(r => r.json())
      .then(data => {
        if (data.data) {
          setModels(data.data);
          
          // Pre-select the most recent d33 and tc models
          const d33s = data.data.filter((m: Record<string, unknown>) => m.target === "d33");
          const tcs = data.data.filter((m: Record<string, unknown>) => m.target === "tc");
          
          if (d33s.length > 0) setSelectedD33(d33s[0].id);
          if (tcs.length > 0) setSelectedTc(tcs[0].id);
        }
      })
      .catch(console.error);
  }, []);

  const handlePredict = async (formula: string) => {
    setIsLoading(true);
    setCompareResults(null);
    
    try {
      const predictRes = await fetch('/api/v1/predict/single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          formula,
          d33_artifact_id: selectedD33 || undefined,
          tc_artifact_id: selectedTc || undefined
        })
      });
      
      let predictData;
      try {
        predictData = await predictRes.json();
      } catch {
        throw new Error("Backend returned an invalid response. Please check if the API server is running.");
      }
      
      if (!predictRes.ok) {
        throw new Error(predictData.detail || "Prediction failed");
      }
      
      const p = predictData.data;

      const hardnessRes = await fetch('/api/v1/hardness/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          formula, 
          predicted_d33: p.predicted_d33, 
          predicted_tc: p.predicted_tc 
        })
      });
      let hardnessData;
      try {
        hardnessData = await hardnessRes.json();
      } catch {
        hardnessData = { data: { vickers_hv: null, mohs: null, use_case: null } };
      }
      const h = hardnessData.data;

      // Extract the top 4 structural features instead of random strings to keep the UI clean
      const topFeatures = Object.entries(p.parsed_features || {})
        .slice(0, 4)
        .reduce((obj, [key, val]) => {
           obj[key] = (val as number).toFixed(3);
           return obj;
        }, {} as Record<string, string>);

      setResult({
        formula: p.formula,
        d33: p.predicted_d33,
        tc: p.predicted_tc,
        d33Lower: p.d33_lower_95,
        d33Upper: p.d33_upper_95,
        tcLower: p.tc_lower_95,
        tcUpper: p.tc_upper_95,
        hardness: {
           vickers: h.vickers_hv,
           mohs: h.mohs,
           use_case: h.use_case
        },
        features: topFeatures
      });
    } catch (e) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to reach Prediction Service. Check Piezo API connection.");
    }
    
    setIsLoading(false);
  };

  const handleComparePredict = async (formulasString: string) => {
    setIsLoading(true);
    setCompareResults(null);
    setResult(null); 
    
    try {
      const formulas = formulasString.split(',').map(f => f.trim());
      const predictRes = await fetch('/api/v1/predict/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ formulas })
      });
      
      const predictData = await predictRes.json();
      
      if (!predictRes.ok) {
        throw new Error(predictData.detail || "Comparison failed");
      }
      
      setCompareResults(predictData.data);
    } catch (e) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to run comparison.");
    }
    
    setIsLoading(false);
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    try {
      const res = await fetch("/api/v1/predict/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ formula: result.formula, prediction_id: "live-session-prediction" })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report_${result.formula as string}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error(e);
    }
  };

  const cards: CardDefinition[] = [
    {
      key: "model-selection",
      title: "Model Ensemble Selection",
      icon: <Cpu className="w-4 h-4 text-primary" />,
      defaultLayout: { x: 0, y: 0, w: 12, h: 3, minW: 6, minH: 2 },
      component: (
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">d33 Prediction Model</label>
              <div className="relative">
                <select value={selectedD33} onChange={(e) => setSelectedD33(e.target.value)} className="w-full appearance-none bg-background border rounded-xl py-3 pl-4 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 cursor-pointer">
                  <option value="">Default System Model</option>
                  {models.filter(m => m.target === "d33").map(m => (
                    <option key={m.id as string} value={m.id as string}>{m.model_name as string} (R² {((m.r2_test as number) || 0).toFixed(3)})</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Tc Prediction Model</label>
              <div className="relative">
                <select value={selectedTc} onChange={(e) => setSelectedTc(e.target.value)} className="w-full appearance-none bg-background border rounded-xl py-3 pl-4 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 cursor-pointer">
                  <option value="">Default System Model</option>
                  {models.filter(m => m.target === "tc").map(m => (
                    <option key={m.id as string} value={m.id as string}>{m.model_name as string} (R² {((m.r2_test as number) || 0).toFixed(3)})</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "formula-input",
      title: "Chemical Formula Input",
      icon: <Droplet className="w-4 h-4 text-blue-500" />,
      defaultLayout: { x: 0, y: 3, w: 12, h: 2, minW: 6, minH: 2 },
      component: (
        <div className="p-4">
          <FormulaInput onPredict={handlePredict} isLoading={isLoading} />
        </div>
      ),
    },
    {
      key: "results",
      title: "Prediction Results",
      icon: <Activity className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 0, y: 5, w: 12, h: 6, minW: 6, minH: 3 },
      component: (
        <div className="p-4">
          <AnimatePresence mode="popLayout">
            {result && !isLoading ? (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">
                <div className="flex justify-end">
                  <Button onClick={handleDownloadReport} variant="outline" className="gap-2">
                    <Download className="w-4 h-4" /> Download Report
                  </Button>
                </div>
                <PredictionGauges d33={result.d33 as number} tc={result.tc as number} d33Lower={result.d33Lower as number} d33Upper={result.d33Upper as number} tcLower={result.tcLower as number} tcUpper={result.tcUpper as number} />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* @ts-expect-error Nested type */}
                  <VickersGauge vickersHv={result.hardness.vickers as number} />
                  {/* @ts-expect-error Nested type */}
                  <MohsScale predictedMohs={result.hardness.mohs as number} />
                </div>
              </motion.div>
            ) : !isLoading && !compareResults ? (
              <div className="flex items-center justify-center flex-col text-muted-foreground/30 py-12">
                <Droplet className="w-12 h-12 mb-3" />
                <p className="text-sm">Awaiting chemical input.</p>
              </div>
            ) : null}
          </AnimatePresence>
        </div>
      ),
    },
    {
      key: "compare",
      title: "Multi-Material Comparison",
      icon: <Zap className="w-4 h-4 text-amber-500" />,
      defaultLayout: { x: 0, y: 11, w: 12, h: 4, minW: 6, minH: 2 },
      component: (
        <div className="p-4 space-y-4">
          <ComparePanel onAddCompare={handleComparePredict} isLoading={isLoading} />
          <AnimatePresence mode="popLayout">
            {compareResults && compareResults.length > 0 && !isLoading && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="rounded-xl border bg-card p-4 overflow-x-auto">
                  <h4 className="font-semibold mb-3">Results</h4>
                  <table className="w-full text-left text-sm">
                    <thead><tr className="border-b"><th className="pb-2 px-3">Formula</th><th className="pb-2 px-3 text-indigo-500">d33</th><th className="pb-2 px-3 text-rose-500">Tc</th></tr></thead>
                    <tbody className="divide-y">
                      {compareResults.map((res, i) => (
                        <tr key={i} className="hover:bg-muted/30"><td className="py-2 px-3 font-mono">{res.formula as string}</td><td className="py-2 px-3 font-bold text-indigo-600">{(res.predicted_d33 as number).toFixed(1)}</td><td className="py-2 px-3 font-bold text-rose-600">{(res.predicted_tc as number).toFixed(1)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ),
    },
    {
      key: "model-consensus",
      title: "Ensemble Consensus Comparison",
      icon: <Shield className="w-4 h-4 text-violet-500" />,
      defaultLayout: { x: 0, y: 15, w: 12, h: 4, minW: 6, minH: 3 },
      component: (
        <div className="p-4">
           {/* @ts-expect-error Nested type */}
           <ModelConsensus currentFormula={result?.formula} models={models} />
        </div>
      ),
    },
    {
      key: "batch-predict",
      title: "Batch Dataset Processing",
      icon: <Database className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 6, y: 11, w: 6, h: 4, minW: 4, minH: 3 },
      component: <BatchPredictPanel />,
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1400px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Live Inference Dashboard</h1>
        <p className="text-muted-foreground">Deploy your active model ensemble against novel chemical mixtures in real-time.</p>
      </div>
      <DraggableGrid pageKey="predict" cards={cards} />
    </div>
  );
}

