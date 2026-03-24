"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

interface InteractiveParetoFrontProps {
  points: {
    id: string;
    d33: number;
    tc: number;
    hardness: number;
    formula: string;
  }[];
}

export function InteractiveParetoFront({ points }: InteractiveParetoFrontProps) {
  return (
    <div className="w-full relative" style={{ minHeight: "500px", height: "500px" }}>
      <div className="absolute top-4 left-4 z-10 bg-black/40 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 text-xs text-white/90 font-mono flex items-center gap-2">
         <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
         2D Projection: Hardness = Bubble Size
      </div>
      
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart
          margin={{
            top: 40,
            right: 40,
            bottom: 40,
            left: 40,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#313244" />
          <XAxis 
            type="number" 
            dataKey="tc" 
            name="Curie Temp" 
            unit="°C" 
            stroke="#a6adc8"
            domain={['dataMin - 20', 'dataMax + 20']}
          />
          <YAxis 
            type="number" 
            dataKey="d33" 
            name="d33" 
            unit=" pC/N" 
            stroke="#a6adc8"
            domain={['dataMin - 50', 'dataMax + 50']}
          />
          <ZAxis type="number" dataKey="hardness" range={[20, 200]} name="Hardness" unit=" HV" />
          
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }} 
            contentStyle={{ backgroundColor: "#1e1e2e", borderColor: "#313244", color: "#cdd6f4", borderRadius: "8px" }}
            itemStyle={{ color: "#cdd6f4" }}
            formatter={((value: unknown, name: string) => {
                const v = Number(value);
                if (name === "Curie Temp" || name === "d33" || name === "Hardness") {
                    return [v.toFixed(1), name];
                }
                return [value as string, name];
            }) as any}
            labelFormatter={() => ""}
          />
          
          <Scatter name="Pareto Front" data={points}>
            {points.map((entry, index) => {
              // Color scale based on hardness directly
              let color = "#89b4fa"; // soft (blue)
              if (entry.hardness > 400) color = "#f9e2af"; // medium (yellow)
              if (entry.hardness > 700) color = "#f38ba8"; // hard (red)
              return <Cell key={`cell-${index}`} fill={color} opacity={0.8} />;
            })}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
