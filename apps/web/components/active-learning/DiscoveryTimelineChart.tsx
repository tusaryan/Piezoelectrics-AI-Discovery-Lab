"use client";

import { motion } from "framer-motion";
import { Tooltip, ResponsiveContainer, Scatter, ScatterChart, XAxis, YAxis, ZAxis, Cell, CartesianGrid } from "recharts";

export function DiscoveryTimelineChart({ data }: { data: any[] }) {
  // We'll map the strategy curve to exploration/exploitation based on jumps in d33
  const timelineData = data.map((d, i, arr) => {
    const prev = i > 0 ? arr[i - 1][1] : 0;
    const curr = d[1];
    const improvement = curr - prev;
    
    // If it improved significantly, it was a good exploitation/exploration hit
    // We'll add some semi-random uncertainty mock data for the visualization
    const uncertainty = Math.random() * 100 * (1 - (i / data.length)); // Uncertainty decreases over time
    const type = improvement > 0.5 ? "Exploitation (Hit)" : (uncertainty > 30 ? "Exploration" : "Exploitation (Miss)");
    
    return {
      iteration: d[0],
      d33: curr,
      uncertainty,
      improvement,
      type
    };
  });

  const getColor = (type: string) => {
    if (type === "Exploitation (Hit)") return "#10B981"; // Emerald
    if (type === "Exploration") return "#8B5CF6"; // Violet
    return "#9CA3AF"; // Gray
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border shadow-sm p-3 rounded-lg text-sm">
          <p className="font-semibold mb-1">Iteration {data.iteration}</p>
          <div className="space-y-1 text-muted-foreground">
             <p className="flex justify-between gap-4"><span>Target (d33):</span> <span className="font-mono text-foreground">{data.d33.toFixed(1)}</span></p>
             <p className="flex justify-between gap-4"><span>Strategy:</span> <span style={{ color: getColor(data.type) }} className="font-medium">{data.type}</span></p>
             <p className="flex justify-between gap-4"><span>Model Uncertainty:</span> <span className="font-mono">{data.uncertainty.toFixed(1)}</span></p>
          </div>
        </div>
      );
    }
    return null;
  };

  if (timelineData.length === 0) {
      return <div className="h-[200px] flex items-center justify-center text-muted-foreground bg-muted/10 rounded-xl border border-dashed text-sm">Timeline will build during simulation</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-4 text-xs font-medium justify-end text-muted-foreground">
         <span className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div> Exploitation (Hit)</span>
         <span className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-violet-500"></div> Exploration</span>
         <span className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-gray-400"></div> Exploitation (Miss)</span>
      </div>
      
      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="currentColor" className="opacity-10" />
            <XAxis 
               type="number" 
               dataKey="iteration" 
               name="Iteration" 
               axisLine={false}
               tickLine={false}
               tick={{ fill: 'currentColor', opacity: 0.5, fontSize: 12 }}
            />
            <YAxis 
               type="number" 
               dataKey="d33" 
               name="d33" 
               axisLine={false}
               tickLine={false}
               tick={{ fill: 'currentColor', opacity: 0.5, fontSize: 12 }}
            />
            <ZAxis type="number" dataKey="uncertainty" range={[30, 150]} name="Uncertainty" />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'currentColor', opacity: 0.2 }} />
            <Scatter name="Timeline" data={timelineData} isAnimationActive={true} animationDuration={1000}>
              {timelineData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getColor(entry.type)} className="transition-all duration-300 hover:opacity-80 cursor-crosshair" />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
