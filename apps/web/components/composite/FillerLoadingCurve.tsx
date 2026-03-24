"use client";

import { useEffect, useState } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { CopyPlus } from "lucide-react";

interface CurvePoint {
  wt_pct: number;
  predicted_d33: number;
  beta_phase_pct: number;
}

interface FillerLoadingCurveProps {
  data: CurvePoint[];
  optimalWt: number;
  currentWt: number;
  onSetWt: (wt: number) => void;
  isLoading: boolean;
}

export function FillerLoadingCurve({ data, optimalWt, currentWt, onSetWt, isLoading }: FillerLoadingCurveProps) {
  const [animatedData, setAnimatedData] = useState<CurvePoint[]>([]);

  useEffect(() => {
    if (!data || data.length === 0) return;
    
    // Animate the line drawing in
    let currentIndex = 0;
    const interval = setInterval(() => {
      currentIndex += 3;
      setAnimatedData(data.slice(0, currentIndex));
      if (currentIndex >= data.length) {
        clearInterval(interval);
        setAnimatedData(data);
      }
    }, 20);

    return () => clearInterval(interval);
  }, [data]);

  if (isLoading || data.length === 0) {
    return (
      <div className="h-[300px] w-full rounded-xl border bg-card flex flex-col items-center justify-center text-muted-foreground shadow-sm">
        <div className="w-8 h-8 rounded-full border-2 border-primary/30 border-t-primary animate-spin mb-4" />
        <p className="text-sm font-medium">Computing Piezoelectric Loading Curve...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border bg-card p-6 shadow-sm w-full relative group">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="font-semibold text-lg flex items-center gap-2">
            <CopyPlus className="w-5 h-5 text-indigo-500" />
            Filler Loading Optimization
          </h3>
          <p className="text-sm text-muted-foreground mt-1">Predicted d₃₃ vs Volumetric/Weight inclusion %</p>
        </div>
        <div className="bg-primary/10 text-primary border border-primary/20 px-3 py-1.5 rounded-lg text-sm font-semibold shadow-sm cursor-pointer hover:bg-primary/20 transition-colors" onClick={() => onSetWt(optimalWt)}>
          Peak: {optimalWt} wt%
        </div>
      </div>

      <div className="h-[280px] w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={animatedData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }} onClick={(e: any) => {
             if (e && e.activePayload && e.activePayload.length > 0) {
                 onSetWt(e.activePayload[0].payload.wt_pct);
             }
          }}>
            <defs>
              <linearGradient id="colorD33" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="currentColor" strokeOpacity={0.1} />
            <XAxis 
              dataKey="wt_pct" 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 12 }} 
              stroke="currentColor" 
              strokeOpacity={0.5} 
            />
            <YAxis 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 12 }} 
              stroke="currentColor" 
              strokeOpacity={0.5}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-black/90 border border-zinc-800 p-3 rounded-lg shadow-xl text-sm">
                      <p className="font-semibold text-zinc-100 mb-1">{label} wt% Filler</p>
                      <p className="text-indigo-400 font-mono">d₃₃: {payload[0].value} pC/N</p>
                      {payload[0].payload.beta_phase_pct && (
                         <p className="text-teal-400 font-mono mt-1 text-xs">PVDF β-Phase: ~{payload[0].payload.beta_phase_pct}%</p>
                      )}
                      <p className="text-zinc-500 text-[10px] mt-2 italic">Click to snap input to {label} wt%</p>
                    </div>
                  );
                }
                return null;
              }}
              cursor={{ stroke: '#6366f1', strokeWidth: 1, strokeDasharray: '4 4' }}
            />
            {animatedData.length > optimalWt && (
              <ReferenceLine x={optimalWt} stroke="#6366f1" strokeDasharray="3 3" label={{ position: 'top', value: 'Optimal Threshold', fill: '#6366f1', fontSize: 10 }} />
            )}
            {animatedData.length > currentWt && (
              <ReferenceLine x={currentWt} stroke="#f59e0b" strokeWidth={2} />
            )}
            <Area 
              type="monotone" 
              dataKey="predicted_d33" 
              stroke="#6366f1" 
              strokeWidth={3}
              fillOpacity={1} 
              fill="url(#colorD33)" 
              animationDuration={0}
              activeDot={{ r: 6, fill: "#6366f1", stroke: "#fff", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
