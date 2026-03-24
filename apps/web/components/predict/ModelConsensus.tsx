"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Activity } from "lucide-react";

export function ModelConsensus({ currentFormula, models }: { currentFormula?: string; models: any[] }) {
  const [selectedModels, setSelectedModels] = useState<string[]>(["", "", ""]);
  const [results, setResults] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleCompare = async () => {
    if (!currentFormula) {
      alert("Please predict a main formula first!");
      return;
    }
    
    const validModels = selectedModels.filter(m => m !== "");
    if (validModels.length === 0) return;

    setIsLoading(true);
    try {
      const preds = await Promise.all(
        validModels.map(async (artifactId) => {
          const m = models.find(x => x.id === artifactId);
          const target = m?.target || "d33";
          
          const payload: any = { formula: currentFormula };
          if (target === "d33") payload.d33_artifact_id = artifactId;
          else payload.tc_artifact_id = artifactId;

          const res = await fetch('/api/v1/predict/single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          return { model: m, result: data.data };
        })
      );
      setResults(preds);
    } catch (e) {
      console.error(e);
      alert("Failed to compare models.");
    }
    setIsLoading(false);
  };

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">Select up to 3 models to evaluate consensus on <strong className="text-foreground">{currentFormula || "the current material"}</strong></div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[0, 1, 2].map(idx => (
          <select 
            key={idx}
            value={selectedModels[idx]} 
            onChange={e => {
              const newModels = [...selectedModels];
              newModels[idx] = e.target.value;
              setSelectedModels(newModels);
            }} 
            className="w-full bg-muted/40 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 appearance-none"
          >
            <option value="">-- Optional Model --</option>
            {models.map(m => (
              <option key={m.id} value={m.id}>{m.model_name} ({m.target.toUpperCase()})</option>
            ))}
          </select>
        ))}
      </div>

      <Button onClick={handleCompare} disabled={isLoading || !currentFormula} className="w-full gap-2">
         <Activity className="w-4 h-4" /> Run Consensus Evaluation
      </Button>

      {results.length > 0 && (
        <div className="mt-4 border rounded-xl overflow-hidden bg-card">
          <table className="w-full text-left text-sm">
            <thead className="bg-muted/30">
              <tr className="border-b">
                <th className="py-2 px-4">Algorithm</th>
                <th className="py-2 px-4">Target</th>
                <th className="py-2 px-4">Prediction</th>
                <th className="py-2 px-4">95% Interval</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {results.map((r, i) => (
                <tr key={i} className="hover:bg-muted/10">
                  <td className="py-3 px-4 font-semibold">{r.model?.model_name}</td>
                  <td className="py-3 px-4 uppercase text-xs text-muted-foreground">{r.model?.target}</td>
                  <td className="py-3 px-4 font-mono font-bold text-primary">
                    {r.model?.target === 'd33' ? r.result.predicted_d33?.toFixed(2) : r.result.predicted_tc?.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 font-mono text-xs text-muted-foreground">
                    [{r.model?.target === 'd33' ? r.result.d33_lower_95?.toFixed(1) : r.result.tc_lower_95?.toFixed(1)} - {r.model?.target === 'd33' ? r.result.d33_upper_95?.toFixed(1) : r.result.tc_upper_95?.toFixed(1)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
