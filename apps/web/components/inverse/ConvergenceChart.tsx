"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface ConvergenceChartProps {
  data: any[];
}

export function ConvergenceChart({ data }: ConvergenceChartProps) {
  return (
    <div className="w-full h-full min-h-[250px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
          <XAxis 
            dataKey="generation" 
            type="number"
            domain={['dataMin', 'dataMax']}
            label={{ value: "Generation", position: "insideBottom", offset: -10, style: { fontSize: 13, fill: 'hsl(var(--muted-foreground))' } }}
            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            dataKey="hypervolume" 
            domain={['auto', 'auto']}
            name="Hypervolume"
            label={{ value: "Hypervolume Indicator", angle: -90, position: "insideLeft", style: { fontSize: 13, fill: 'hsl(var(--muted-foreground))' } }}
            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip 
             contentStyle={{ backgroundColor: 'hsl(var(--popover))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
             itemStyle={{ color: 'hsl(var(--foreground))' }}
             labelStyle={{ color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}
             formatter={(value: any) => [Number(value || 0).toFixed(2), "Hypervolume"]}
             labelFormatter={(label) => `Generation ${label}`}
          />
          <Line 
            type="monotone" 
            dataKey="hypervolume" 
            stroke="#10B981" 
            strokeWidth={3}
            dot={false}
            activeDot={{ r: 6, fill: "#10B981", stroke: "#fff", strokeWidth: 2 }}
            isAnimationActive={false} // Disable default animation because we add points progressively
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
