"use client";

import { useEffect, useState } from "react";
import { Network } from "lucide-react";
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";

export function GraphEmbeddingsCard() {
  const [data, setData] = useState<{ nodes: any[], edges: any[], formula: string, similarity_to_pzt: number } | null>(null);

  useEffect(() => {
    fetch("/api/v1/interpret/embeddings?formula=BaTiO3")
      .then(res => res.json())
      .then(d => {
        if (d.success) {
          setData(d.data);
        }
      })
      .catch(e => console.error(e));
  }, []);

  return (
    <div className="rounded-xl border bg-card shadow-sm overflow-hidden h-full flex flex-col">
       <div className="p-6 border-b bg-muted/20">
          <h3 className="font-semibold text-lg flex items-center gap-2">
             <Network className="w-5 h-5 text-sky-500" />
             Latent Structural Embeddings (CHGNet)
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            2D projection of graph neural network node activations, evaluating crystallographic motif similarity against benchmark structures like PZT.
          </p>
       </div>
       <div className="p-6 flex-1 flex flex-col items-center justify-center">
         {data ? (
            <div className="w-full h-full flex flex-col gap-4 min-h-0">
               <div className="flex justify-between items-center bg-[#0D0E19] px-4 py-3 border border-white/5 rounded-lg shrink-0">
                  <div className="text-sm text-muted-foreground">Target Scaffold</div>
                  <div className="font-mono text-emerald-400 font-bold tracking-wider">{data.formula}</div>
                  <div className="text-sm text-muted-foreground">Similarity Score (vs PbTiO3)</div>
                  <div className="font-mono text-white text-lg font-bold">{(data.similarity_to_pzt * 100).toFixed(1)}%</div>
               </div>
               
               <div className="relative flex-1 bg-[#11111B] rounded-lg border border-white/5 overflow-hidden min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                     <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#313244" />
                        <XAxis type="number" dataKey="x" name="Latent Dim 1" stroke="#a6adc8" />
                        <YAxis type="number" dataKey="y" name="Latent Dim 2" stroke="#a6adc8" />
                        <ZAxis type="number" dataKey="latent_activation" range={[50, 400]} name="Activation Heat" />
                        <Tooltip 
                           cursor={{ strokeDasharray: '3 3' }}
                           contentStyle={{ backgroundColor: "#1e1e2e", borderColor: "#313244", color: "#cdd6f4", borderRadius: "8px" }}
                        />
                        <Scatter name="Atoms" data={data.nodes} fill="#89b4fa" opacity={0.6}>
                        </Scatter>
                     </ScatterChart>
                  </ResponsiveContainer>
                  
                  <div className="absolute bottom-3 right-3 text-[10px] text-muted-foreground font-mono">
                     T-SNE Dimensionality Reduction • Graph Size: {data.nodes.length} nodes, {data.edges.length} edges
                  </div>
               </div>
            </div>
         ) : (
            <div className="animate-pulse text-muted-foreground flex flex-col items-center">
               <div className="w-8 h-8 border-2 border-t-sky-500 border-sky-500/20 rounded-full animate-spin mb-4" />
               Extracting ALIGNN representations...
            </div>
         )}
       </div>
    </div>
  );
}
