"use client";

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface ComplexityAccuracyPlotProps {
  data: any[];
  onDotClick?: (equation: any) => void;
  selectedEquationId?: string; // We can use equation string as ID
}

export function ComplexityAccuracyPlot({ data, onDotClick, selectedEquationId }: ComplexityAccuracyPlotProps) {
  
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      return (
        <div className="bg-popover border text-popover-foreground p-3 rounded-md shadow-md text-sm max-w-xs">
          <p className="font-semibold mb-1">Complexity: {point.complexity}</p>
          <p className="mb-1">R²: <span className="text-emerald-500 font-mono">{point.r2.toFixed(3)}</span></p>
          <div className="mt-2 pt-2 border-t font-mono text-xs overflow-hidden text-ellipsis whitespace-nowrap text-muted-foreground">
             {point.equation}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full min-h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
          <XAxis 
            type="number" 
            dataKey="complexity" 
            name="Complexity" 
            label={{ value: "Equation Complexity", position: "insideBottom", offset: -10, style: { fill: 'hsl(var(--muted-foreground))', fontSize: 13 } }}
            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(var(--border))' }}
          />
          <YAxis 
            type="number" 
            dataKey="r2" 
            name="R² Score" 
            domain={['auto', 'auto']}
            label={{ value: "R² Score", angle: -90, position: "insideLeft", style: { fill: 'hsl(var(--muted-foreground))', fontSize: 13 } }}
            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(var(--border))' }}
            tickFormatter={(val) => val.toFixed(2)}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'hsl(var(--muted))' }} />
          <Scatter 
            name="Equations" 
            data={data} 
            fill="#4F46E5"
            onClick={(e) => onDotClick && onDotClick(e)}
            shape={(props: any) => {
                const { cx, cy, payload } = props;
                const isSelected = payload.equation === selectedEquationId;
                return (
                    <circle 
                        cx={cx} 
                        cy={cy} 
                        r={isSelected ? 8 : 5} 
                        fill={isSelected ? "#10B981" : "#4F46E5"} 
                        stroke={isSelected ? "#fff" : "none"}
                        strokeWidth={2}
                        className="transition-all duration-300 ease-in-out hover:opacity-80 cursor-pointer"
                    />
                );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
