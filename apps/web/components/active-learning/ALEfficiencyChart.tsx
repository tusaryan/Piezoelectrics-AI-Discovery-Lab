"use client";

import { motion } from "framer-motion";
import { Area, AreaChart, CartesianGrid, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis, Line, ComposedChart } from "recharts";

export function ALEfficiencyChart({ 
    strategyData, 
    baselineData, 
    efficiencyGain 
}: { 
    strategyData: any[]; 
    baselineData: any[]; 
    efficiencyGain: number 
}) {
  // Combine data for ComposedChart
  const combined = strategyData.map((d, i) => ({
    iteration: d[0],
    strategy_d33: d[1],
    baseline_d33: baselineData[i] ? baselineData[i][1] : null,
    // Add shading area
    area_min: baselineData[i] ? Math.min(d[1], baselineData[i][1]) : null,
    area_max: Math.max(d[1], baselineData[i] ? baselineData[i][1] : 0)
  }));

  if (combined.length === 0) {
     return <div className="h-[400px] flex items-center justify-center text-muted-foreground bg-muted/10 rounded-xl border border-dashed">Run simulation to view learning curve</div>;
  }

  return (
    <div className="h-[400px] w-full">
      <div className="absolute top-6 right-6 z-10">
        {efficiencyGain > 0 && (
          <motion.div 
             initial={{ scale: 0.8, opacity: 0 }}
             animate={{ scale: 1, opacity: 1 }}
             className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-3 py-1.5 rounded-full text-sm font-semibold border border-emerald-500/20"
          >
            {efficiencyGain}% fewer experiments needed
          </motion.div>
        )}
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={combined} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="efficiencyArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.2}/>
              <stop offset="95%" stopColor="#4F46E5" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="currentColor" className="opacity-10" />
          <XAxis 
             dataKey="iteration" 
             axisLine={false}
             tickLine={false}
             tick={{ fill: 'currentColor', opacity: 0.5, fontSize: 12 }}
             dy={10}
             label={{ value: "Experiments Conducted", position: 'insideBottom', offset: -10, fill: "currentColor", opacity: 0.5, fontSize: 12 }}
          />
          <YAxis 
             domain={['auto', 'auto']}
             axisLine={false}
             tickLine={false}
             tick={{ fill: 'currentColor', opacity: 0.5, fontSize: 12 }}
             dx={-10}
             label={{ value: "Best d33 (pC/N)", angle: -90, position: 'insideLeft', fill: "currentColor", opacity: 0.5, fontSize: 12 }}
          />
          <Tooltip 
             contentStyle={{ borderRadius: '8px', border: '1px solid var(--border-color, #E4E6F1)', backgroundColor: 'var(--bg-surface, #FFFFFF)' }}
             labelStyle={{ color: 'var(--text-muted, #9CA3AF)', marginBottom: '4px' }}
             itemStyle={{ fontSize: '13px', fontWeight: 500 }}
             formatter={((value: any, name?: string) => [value, name === 'strategy_d33' ? 'AI Strategy' : 'Random Baseline']) as any}
             labelFormatter={(l) => `Iteration ${l}`}
          />
          
          {/* Shaded Area between curves */}
          <Area 
             type="stepAfter" 
             dataKey="strategy_d33" 
             fill="url(#efficiencyArea)" 
             stroke="none" 
             activeDot={false}
             isAnimationActive={true}
             animationDuration={1500}
          />
          
          <Line 
             type="stepAfter" 
             dataKey="baseline_d33" 
             stroke="#9CA3AF" 
             strokeWidth={2} 
             strokeDasharray="5 5"
             dot={false}
             isAnimationActive={true}
             animationDuration={1500}
          />
          <Line 
             type="stepAfter" 
             dataKey="strategy_d33" 
             stroke="#4F46E5" 
             strokeWidth={3} 
             dot={false}
             activeDot={{ r: 6, fill: "#4F46E5", stroke: "white", strokeWidth: 2 }}
             isAnimationActive={true}
             animationDuration={1500}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
