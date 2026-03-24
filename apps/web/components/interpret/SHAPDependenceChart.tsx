"use client";

import { useState, useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ZAxis,
  Cell,
} from "recharts";

interface GlobalShapFeature {
  feature: string;
  mean_abs_shap: number;
  shap_values: number[];
  feature_values: number[];
}

export function SHAPDependenceChart({ data }: { data: GlobalShapFeature[] }) {
  const [selectedFeature, setSelectedFeature] = useState<string>(
    data[0]?.feature || ""
  );

  const chartData = useMemo(() => {
    const feat = data.find((d) => d.feature === selectedFeature);
    if (!feat) return [];

    const pts = [];
    for (let i = 0; i < feat.shap_values.length; i++) {
      pts.push({
        x: feat.feature_values[i],
        y: feat.shap_values[i],
      });
    }
    return pts;
  }, [data, selectedFeature]);

  if (!data || data.length === 0) return null;

  return (
    <div className="flex flex-col h-full space-y-2">
      <div className="flex items-center justify-between text-sm px-2 mt-1">
        <span className="text-muted-foreground font-medium">Select Feature:</span>
        <select
          value={selectedFeature}
          onChange={(e) => setSelectedFeature(e.target.value)}
          className="bg-muted/30 border border-border rounded-lg px-2 py-1 text-xs focus:ring-1 focus:ring-primary/50 text-foreground"
        >
          {data.slice(0, 10).map((d) => (
            <option key={d.feature} value={d.feature}>
              {d.feature}
            </option>
          ))}
        </select>
      </div>

      <div className="flex-1 min-h-[250px] w-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
            <XAxis
              type="number"
              dataKey="x"
              name="Feature Value"
              tick={{ fontSize: 11, fill: "#888" }}
              axisLine={{ stroke: "#333" }}
              tickLine={false}
              domain={["auto", "auto"]}
            >
            </XAxis>
            <YAxis
              type="number"
              dataKey="y"
              name="SHAP Value"
              tick={{ fontSize: 11, fill: "#888" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <ZAxis type="number" range={[20, 20]} />
            <RechartsTooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-popover border border-border p-3 rounded-lg shadow-xl text-xs">
                      <p className="font-semibold text-foreground mb-1">
                        {selectedFeature}
                      </p>
                      <div className="space-y-1">
                        <div className="flex justify-between gap-4">
                          <span className="text-muted-foreground">Value:</span>
                          <span className="font-mono text-indigo-400">
                            {data.x.toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between gap-4">
                          <span className="text-muted-foreground">Impact on Output:</span>
                          <span
                            className={data.y > 0 ? "text-rose-400 font-mono" : "text-blue-400 font-mono"}
                          >
                            {data.y > 0 ? "+" : ""}
                            {data.y.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            {/* Horizontal line at y=0 */}
            <Scatter name="SHAP" data={chartData} fill="#8884d8">
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.y > 0 ? "rgba(244, 63, 94, 0.7)" : "rgba(59, 130, 246, 0.7)"}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[10px] text-muted-foreground text-center px-4">
        X-axis: Actual feature value. Y-axis: SHAP contribution to prediction.
      </p>
    </div>
  );
}
