"use client";

import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface ParetoFrontChartProps {
  data: any[];
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onClick: (id: string) => void;
}

export function ParetoFrontChart({ data, hoveredId, onHover, onClick }: ParetoFrontChartProps) {
  
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const sol = payload[0].payload;
      return (
        <div className="bg-popover border text-popover-foreground p-3 rounded-md shadow-md text-sm z-50 pointer-events-none">
          <p className="font-bold border-b pb-1 mb-2 text-xs uppercase text-indigo-500">{sol.use_case}</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
             <span className="text-muted-foreground">Predicted d33:</span>
             <span className="font-mono text-indigo-600 font-semibold">{sol.predicted_d33?.toFixed(1)}</span>
             <span className="text-muted-foreground">Predicted Tc:</span>
             <span className="font-mono text-emerald-600 font-semibold">{sol.predicted_tc?.toFixed(1)}</span>
             <span className="text-muted-foreground">Hardness:</span>
             <span className="font-mono text-amber-600 font-semibold">{sol.predicted_hardness?.toFixed(2)}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full min-h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          
          <XAxis 
            type="number" 
            dataKey="predicted_d33" 
            name="Predicted d33" 
            label={{ value: "Predicted d33 (pC/N)", position: "insideBottom", offset: -10, style: { fontSize: 13, fill: 'hsl(var(--muted-foreground))' } }}
            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
          />
          <YAxis 
            type="number" 
            dataKey="predicted_tc" 
            name="Predicted Tc" 
            label={{ value: "Predicted Tc (°C)", angle: -90, position: "insideLeft", style: { fontSize: 13, fill: 'hsl(var(--muted-foreground))' } }}
            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
          />
          <ZAxis type="number" dataKey="predicted_hardness" range={[60, 200]} name="Hardness" />
          
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          
          <Scatter 
            name="Solutions" 
            data={data} 
            onMouseMove={(e: any) => e?.payload && onHover(e.payload.id)}
            onMouseLeave={() => onHover(null)}
            onClick={(e: any) => e?.payload && onClick(e.payload.id)}
            shape={(props: any) => {
                const { cx, cy, payload } = props;
                // Color mapping logic for Hardness (amber to red)
                const v = payload.predicted_hardness;
                // Min ~2, Max ~10
                const norm = Math.max(0, Math.min(1, (v - 2) / 8));
                const r = Math.round(245 - norm * 50);
                const g = Math.round(158 - norm * 158);
                const b = Math.round(11 - norm * 11);
                const fill = `rgb(${r},${g},${b})`;
                
                const isHovered = payload.id === hoveredId;
                const isDimmed = hoveredId !== null && !isHovered;
                
                return (
                    <circle 
                        cx={cx} 
                        cy={cy} 
                        r={isHovered ? 8 : 6} 
                        fill={fill} 
                        stroke={isHovered ? "#fff" : "none"}
                        strokeWidth={2}
                        opacity={isDimmed ? 0.2 : 0.8}
                        className="transition-all duration-200 cursor-pointer"
                    />
                );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
