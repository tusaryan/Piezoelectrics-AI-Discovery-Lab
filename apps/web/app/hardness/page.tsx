"use client";

import { useState } from "react";
import { Diamond, Search, Activity, Cpu, Box, LayoutPanelTop } from "lucide-react";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

export default function HardnessPage() {
  const [formData, setFormData] = useState({
    formula: "Pb(Zr,Ti)O3",
    predicted_d33: 400,
    predicted_tc: 350,
    matrix_type: ""
  });

  interface UseCase {
    use_case: string;
    description: string;
    recommended_applications: string[];
    icon: string;
    confidence: number;
  }

  interface HardnessResult {
    vickers_hv: number;
    mohs: number;
    use_case: UseCase;
  }

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<HardnessResult | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/hardness/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          matrix_type: formData.matrix_type === "" ? null : formData.matrix_type
        })
      });
      const data = await res.json();
      if (data.status === "success") setResult(data.data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const cards: CardDefinition[] = [
    {
      key: "input",
      title: "Material Properties",
      icon: <LayoutPanelTop className="w-4 h-4 text-blue-500" />,
      defaultLayout: { x: 0, y: 0, w: 4, h: 6, minW: 3, minH: 4 },
      component: (
        <div className="p-6 space-y-4">
          <div>
            <label className="text-xs font-semibold text-muted-foreground uppercase mb-1.5 block">Chemical Formula</label>
            <input type="text" value={formData.formula} onChange={(e) => setFormData({...formData, formula: e.target.value})} className="w-full h-10 px-3 rounded-lg border bg-background text-sm font-mono" placeholder="e.g. BaTiO3" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-semibold text-muted-foreground uppercase mb-1.5 block">Est. d₃₃ (pC/N)</label>
              <input type="number" value={formData.predicted_d33} onChange={(e) => setFormData({...formData, predicted_d33: parseFloat(e.target.value) || 0})} className="w-full h-10 px-3 rounded-lg border bg-background text-sm font-mono" />
            </div>
            <div>
              <label className="text-xs font-semibold text-muted-foreground uppercase mb-1.5 block">Est. Tc (°C)</label>
              <input type="number" value={formData.predicted_tc} onChange={(e) => setFormData({...formData, predicted_tc: parseFloat(e.target.value) || 0})} className="w-full h-10 px-3 rounded-lg border bg-background text-sm font-mono" />
            </div>
          </div>
          <div className="pt-4 border-t mt-2">
            <label className="text-xs font-semibold text-muted-foreground uppercase mb-1.5 block">Polymer Matrix (Optional)</label>
            <select value={formData.matrix_type} onChange={(e) => setFormData({...formData, matrix_type: e.target.value})} className="w-full h-10 px-3 rounded-lg border bg-background text-sm">
              <option value="">None (Bulk Ceramic)</option>
              <option value="pvdf">PVDF</option>
              <option value="epoxy">Epoxy Resin</option>
              <option value="silicone">Silicone Elastomer</option>
            </select>
          </div>
          <button onClick={handlePredict} disabled={loading || !formData.formula} className="w-full mt-4 h-10 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium flex items-center justify-center gap-2 transition-colors disabled:opacity-50">
            {loading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <><Search className="w-4 h-4" />Analyze Mechanics</>}
          </button>
        </div>
      ),
    },
    {
      key: "vickers",
      title: "Vickers Hardness",
      icon: <Box className="w-4 h-4 text-blue-500" />,
      defaultLayout: { x: 4, y: 0, w: 4, h: 3, minW: 3, minH: 2 },
      component: (
        <div className="p-6">
          {result ? (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-5xl font-bold tracking-tighter text-blue-500">{result.vickers_hv.toFixed(1)}</span>
                <span className="text-sm text-muted-foreground font-mono">HV</span>
              </div>
              <div className="mt-4 text-xs text-muted-foreground border-t pt-3">
                {result.vickers_hv < 100 ? "Extremely soft / flexible." : result.vickers_hv > 800 ? "Highly dense rigid ceramic." : "Standard piezoceramic durability."}
              </div>
            </>
          ) : (
            <div className="h-[80px] flex items-center justify-center text-muted-foreground">Enter material to predict</div>
          )}
        </div>
      ),
    },
    {
      key: "mohs",
      title: "Mohs Scale",
      icon: <Activity className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 8, y: 0, w: 4, h: 3, minW: 3, minH: 2 },
      component: (
        <div className="p-6">
          {result ? (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-5xl font-bold tracking-tighter text-indigo-500">{result.mohs.toFixed(1)}</span>
                <span className="text-sm text-muted-foreground font-mono">/ 10</span>
              </div>
              <div className="mt-4 text-xs text-muted-foreground border-t pt-3">
                Scratch resistance equivalent to naturally occurring minerals.
              </div>
            </>
          ) : (
            <div className="h-[80px] flex items-center justify-center text-muted-foreground">Enter material to predict</div>
          )}
        </div>
      ),
    },
    {
      key: "use-case",
      title: "Commercial Domain Mapping",
      icon: <Cpu className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 4, y: 3, w: 8, h: 3, minW: 5, minH: 2 },
      component: (
        <div className="p-6">
          {result ? (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">{result.use_case.use_case}</h2>
                <span className="text-xs font-mono bg-emerald-500/10 text-emerald-500 py-1 px-2 rounded-md font-bold">
                  {result.use_case.confidence * 100}% CONFIDENCE
                </span>
              </div>
              <p className="text-muted-foreground text-sm mb-4">{result.use_case.description}</p>
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Recommended Applications</h4>
                <div className="flex flex-wrap gap-2">
                  {result.use_case.recommended_applications.map((app: string, idx: number) => (
                    <span key={idx} className="bg-secondary text-secondary-foreground px-3 py-1.5 rounded-full text-xs font-medium">{app}</span>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="h-[120px] flex flex-col items-center justify-center text-muted-foreground">
              <Diamond className="w-8 h-8 opacity-30 mb-2" />
              <p className="text-sm">Enter material properties to map commercial domains.</p>
            </div>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1400px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-3">
          <Diamond className="w-8 h-8 text-blue-500" />
          Mechanical Informatics
        </h1>
        <p className="text-muted-foreground">
          Predict Vickers and Mohs hardness for piezoelectric materials and map durability to commercial use cases.
        </p>
      </div>
      <DraggableGrid pageKey="hardness" cards={cards} />
    </div>
  );
}
