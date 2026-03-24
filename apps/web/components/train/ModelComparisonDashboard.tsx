"use client";

import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';
import { Layers, CheckCircle2, Circle, Maximize2, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Artifact {
  id: string;
  target: string;
  model_name: string;
  r2_test: number | null;
  rmse_test: number | null;
  created_at: string;
}

const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#ef4444', '#14b8a6', '#6366f1'];

export function ModelComparisonDashboard({ artifacts }: { artifacts: Artifact[] }) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [fullscreenGraph, setFullscreenGraph] = useState<{ target: string, type: 'R2' | 'RMSE' } | null>(null);

  // Sort by created_at descending (newest first)
  const sortedArtifacts = [...artifacts].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  // Default to selecting the most recent 5 models when artifacts arrive
  useEffect(() => {
    if (artifacts.length > 0 && selectedIds.size === 0) {
      const initialSelection = new Set(sortedArtifacts.slice(0, 5).map(a => a.id));
      setSelectedIds(initialSelection);
    }
  }, [artifacts, selectedIds.size, sortedArtifacts]); // Added sortedArtifacts to dependencies

  const toggleSelection = (id: string) => {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedIds(next);
  };

  if (artifacts.length === 0) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center text-muted-foreground border border-dashed rounded-xl bg-muted/20 p-8 text-sm">
        <Layers className="w-8 h-8 mb-4 opacity-50" />
        <p>No completed models found for this dataset.</p>
        <p className="opacity-75 mt-1">Initialize a pipeline run to generate comparison verification metrics.</p>
      </div>
    );
  }

  const selectedArtifacts = sortedArtifacts.filter(a => selectedIds.has(a.id));
  const activeTargets = Array.from(new Set(selectedArtifacts.map(a => a.target)));

  const renderChart = (data: any[], type: 'R2' | 'RMSE', inFullscreen = false) => {
     const isR2 = type === 'R2';
     const title = isR2 ? "Metric Comparison: Accuracy (R²)" : "Metric Comparison: Error (RMSE)";
     const subtitle = isR2 ? `Testing Set R² Score (Higher is better)` : `Testing Set RMSE (Lower is better)`;
     const labelSize = inFullscreen ? 14 : 10;
     const tickFill = '#71717a';
     
     return (
       <div className={`flex flex-col ${inFullscreen ? 'h-full w-full' : 'h-[320px] relative group'}`}>
          {!inFullscreen && (
             <button 
                onClick={() => setFullscreenGraph({ target: data[0]?.rawTarget, type })}
                className="absolute top-0 right-0 p-1.5 bg-background/80 hover:bg-muted rounded text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity z-10 border shadow-sm"
             >
                <Maximize2 className="w-4 h-4" />
             </button>
          )}
          
          <div className="text-center mb-4">
             <h4 className={`font-bold text-foreground ${inFullscreen ? 'text-2xl mb-2' : 'text-sm'}`}>{title}</h4>
             <p className={`text-muted-foreground ${inFullscreen ? 'text-base' : 'text-xs'}`}>{subtitle}</p>
          </div>
          
          <div className="flex-1 w-full min-h-0">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: inFullscreen ? 80 : 60 }}>
                 <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--muted-foreground)/0.15)" />
                 <XAxis 
                   dataKey="name" 
                   tick={{ fontSize: labelSize, fill: tickFill }} 
                   tickLine={false} 
                   axisLine={{ stroke: 'hsl(var(--muted-foreground)/0.3)' }}
                   angle={-35}
                   textAnchor="end"
                 />
                 <YAxis 
                   tick={{ fontSize: labelSize, fill: tickFill }} 
                   tickLine={false} 
                   axisLine={false} 
                   domain={['auto', 'auto']} 
                   width={60} 
                 />
                 <Tooltip 
                   cursor={{ fill: 'hsl(var(--muted)/0.4)' }} 
                   contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', color: 'hsl(var(--foreground))', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} 
                 />
                 <Bar dataKey={type} radius={[4, 4, 0, 0]} minPointSize={2}>
                   {data.map((entry, index) => {
                      // V1 Styling: Blue for d33 R2, Green for Tc R2, Red for RMSE
                      let barColor = entry.color; 
                      if (type === 'RMSE') barColor = '#ef4444'; // Red
                      else if (entry.rawTarget === 'd33') barColor = '#3b82f6'; // Blue
                      else if (entry.rawTarget === 'tc') barColor = '#10b981'; // Green
                      
                      return <Cell key={`cell-${entry.uid}-${index}`} fill={barColor} fillOpacity={0.9} />;
                   })}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>
       </div>
     );
  };

  return (
    <>
      <div className="flex flex-col gap-4 h-full min-h-[500px]">
        
        {/* Selection Filter UI */}
        <div className="bg-muted/10 p-4 rounded-xl border border-border shrink-0">
           <h4 className="text-xs font-bold text-muted-foreground uppercase mb-3 tracking-wider">Include in Comparison</h4>
           <div className="flex flex-wrap gap-2 max-h-[120px] overflow-y-auto pr-2 pb-1">
              {sortedArtifacts.map(art => {
                 const isSelected = selectedIds.has(art.id);
                 const dateLabel = new Date(art.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                 return (
                    <button 
                       key={art.id}
                       onClick={() => toggleSelection(art.id)}
                       className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${
                          isSelected 
                            ? 'border-primary bg-primary/10 text-primary font-bold shadow-sm' 
                            : 'border-border bg-card text-muted-foreground hover:border-primary/50'
                       }`}
                    >
                       {isSelected ? <CheckCircle2 className="w-4 h-4" /> : <Circle className="w-4 h-4" />}
                       <span>{art.model_name} <span className="opacity-60 text-[10px] ml-1">[{art.target}]</span></span>
                    </button>
                 );
              })}
           </div>
        </div>
        
        {/* Graphs Container */}
        <div className="flex-1 overflow-y-auto pr-2 space-y-8 pb-4">
          {selectedArtifacts.length === 0 ? (
             <div className="py-12 text-center text-sm text-muted-foreground">Select at least one model above to view metrics.</div>
          ) : activeTargets.map(target => {
            const targetData = selectedArtifacts.filter(a => a.target === target).map((art, idx) => ({
               name: art.model_name,
               uid: art.id.substring(0, 5),
               rawTarget: target,
               R2: art.r2_test ? parseFloat(art.r2_test.toFixed(3)) : 0,
               RMSE: art.rmse_test ? parseFloat(art.rmse_test.toFixed(3)) : 0,
               color: COLORS[idx % COLORS.length]
            }));

            return (
              <div key={target} className="bg-card p-6 rounded-2xl border border-border shadow-sm">
                <h4 className="text-sm font-bold text-foreground uppercase mb-6 tracking-wider flex justify-between items-center border-b pb-3">
                  <span className="flex items-center gap-2">
                     <div className="w-2 h-2 rounded-full bg-primary" /> Target: {target}
                  </span>
                  <span className="text-xs normal-case font-medium text-muted-foreground bg-muted px-2 py-1 rounded-md">{targetData.length} models</span>
                </h4>
                
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                  {renderChart(targetData, 'R2')}
                  {renderChart(targetData, 'RMSE')}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Fullscreen Graph Modal */}
      <AnimatePresence>
         {fullscreenGraph && (
            <motion.div 
               initial={{ opacity: 0 }}
               animate={{ opacity: 1 }}
               exit={{ opacity: 0 }}
               className="fixed inset-0 z-[100] bg-background/95 backdrop-blur-md p-6 md:p-12 flex flex-col"
            >
               <div className="flex justify-between items-center mb-8 shrink-0 border-b pb-4">
                   <div className="flex items-center gap-3">
                      <Layers className="w-6 h-6 text-primary" />
                      <h2 className="text-2xl font-bold">Detailed Analysis View</h2>
                   </div>
                   <button 
                      onClick={() => setFullscreenGraph(null)}
                      className="p-2 bg-muted hover:bg-muted/80 rounded-full transition-colors"
                   >
                      <X className="w-6 h-6" />
                   </button>
               </div>
               
               <div className="flex-1 w-full max-w-6xl mx-auto flex items-center justify-center p-8 bg-card rounded-2xl shadow-lg border border-border">
                  {(() => {
                     const tData = selectedArtifacts
                        .filter(a => a.target === fullscreenGraph.target)
                        .map((art, idx) => ({
                           name: art.model_name,
                           uid: art.id.substring(0, 5),
                           rawTarget: fullscreenGraph.target,
                           R2: art.r2_test ? parseFloat(art.r2_test.toFixed(3)) : 0,
                           RMSE: art.rmse_test ? parseFloat(art.rmse_test.toFixed(3)) : 0,
                           color: COLORS[idx % COLORS.length]
                        }));
                     return renderChart(tData, fullscreenGraph.type, true);
                  })()}
               </div>
            </motion.div>
         )}
      </AnimatePresence>
    </>
  );
}
