"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export function TrainingLossChart({ data }: { data: { epoch: number, loss: number }[] }) {
  return (
    <div className="h-[200px] w-full mt-4">
      {data.length === 0 ? (
        <div className="h-full w-full flex items-center justify-center text-muted-foreground text-sm border-2 border-dashed rounded-xl">
          Awaiting ML epoch iterations...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--muted-foreground)/0.2)" />
            <XAxis 
              dataKey="epoch" 
              tick={{ fontSize: 10 }} 
              tickLine={false} 
              axisLine={false} 
              domain={['dataMin', 'dataMax']} // Fixes x-axis stacking issues
            />
            <YAxis 
              domain={['auto', 'auto']} 
              tick={{ fontSize: 10 }} 
              tickLine={false} 
              axisLine={false} 
              width={50} 
            />
            <Tooltip 
              contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
              labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
              formatter={(val: any) => Number(val).toFixed(4)}
            />
            <Line 
              type="monotone" 
              dataKey="loss" 
              stroke="#10b981" 
              strokeWidth={2} 
              dot={{ r: 2, fill: '#10b981' }}
              isAnimationActive={false} 
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
