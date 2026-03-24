"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, ReferenceLine } from "recharts";

interface WaterfallPoint {
  feature: string;
  value: string;
  shap_contribution: number;
  cumulative: number;
}

interface SHAPWaterfallProps {
  data: WaterfallPoint[];
  baseline: number;
}

export function SHAPWaterfallChart({ data, baseline }: SHAPWaterfallProps) {
  // Transform data for Recharts stacked/waterfall logic
  const chartData = data.map((d, i) => {
     const isPositive = d.shap_contribution > 0;
     const start = i === 0 ? baseline : data[i-1].cumulative;
     const end = d.cumulative;
     
     return {
         name: d.feature,
         start: Math.min(start, end),
         end: Math.max(start, end),
         val: d.value,
         impact: d.shap_contribution,
         isPositive
     };
  });
  
  // Add final prediction bar
  if (chartData.length > 0) {
      const finalVal = data[data.length-1].cumulative;
      chartData.push({
          name: "Final Prediction",
          start: 0,
          end: finalVal,
          val: "Output",
          impact: finalVal - baseline,
          isPositive: finalVal > baseline
      });
  }

  return (
    <div className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#374151" strokeOpacity={0.2} />
          <XAxis type="number" domain={['auto', 'auto']} tick={{fontSize: 12}} />
          <YAxis 
            dataKey="name" 
            type="category" 
            axisLine={false} 
            tickLine={false} 
            tick={{fontSize: 11}} 
            width={110}
          />
          <Tooltip 
             cursor={{fill: 'transparent'}}
             content={({ active, payload }) => {
                if (active && payload && payload.length) {
                   const d = payload[0].payload;
                   return (
                     <div className="bg-black/90 border border-zinc-800 p-3 rounded-lg shadow-xl">
                        <p className="text-sm font-semibold text-zinc-200">{d.name}</p>
                        <p className="text-xs text-zinc-400 mt-1">Value: {d.val}</p>
                        <p className={`text-sm font-mono mt-2 ${d.isPositive ? 'text-rose-400' : 'text-blue-400'}`}>
                           Impact: {d.impact > 0 ? '+' : ''}{d.impact.toFixed(2)}
                        </p>
                     </div>
                   )
                }
                return null;
             }}
          />
          <ReferenceLine x={baseline} stroke="#9ca3af" strokeDasharray="4 4" />
          <Bar dataKey="end" fill="#8884d8" barSize={20}>
            {chartData.map((entry, index) => {
              if (index === chartData.length - 1) return <Cell key={index} fill="#8b5cf6" />; // Final
              return <Cell key={index} fill={entry.isPositive ? "#f43f5e" : "#3b82f6"} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
