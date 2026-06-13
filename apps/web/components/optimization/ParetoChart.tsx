"use client";

import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useOptimizationStore } from "@/lib/store/optimizationStore";
import ChartNavigator from "@/components/ui/ChartNavigator";

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃ (pC/N)",
  tc: "Tc (°C)",
  vickers_hardness: "Hardness (HV)",
};

const USE_CASE_COLORS: Record<string, string> = {
  "Flexible Wearable": "#3B82F6",
  "Energy Harvesting": "#10B981",
  "Industrial Actuator": "#F59E0B",
  "Ultrasonic Transducer": "#EF4444",
  "High-Temp Sensor": "#8B5CF6",
  "Sonar/Underwater": "#06B6D4",
  "General Purpose": "#6B7280",
};

export default function ParetoChart() {
  const { solutions, optimizationStats } = useOptimizationStore();
  const [xAxis, setXAxis] = useState("d33");
  const [yAxis, setYAxis] = useState("tc");

  const targets = optimizationStats?.targets_optimized || [];

  const data = useMemo(() => {
    return solutions.map((s) => ({
      x: s.predicted[xAxis] ?? 0,
      y: s.predicted[yAxis] ?? 0,
      formula: s.formula_approx,
      tag: s.use_case_tag,
      color: s.use_case_color || USE_CASE_COLORS[s.use_case_tag] || "#6B7280",
      rank: s.rank,
      ...s.predicted,
    }));
  }, [solutions, xAxis, yAxis]);

  if (solutions.length === 0) return null;

  // Unique use-case tags for legend
  const uniqueTags = [...new Set(solutions.map((s) => s.use_case_tag))];

  return (
    <div className="opt-card opt-chart-card">
      <div className="opt-card-header">
        <h3>Pareto Front</h3>
        <div className="opt-chart-controls">
          <div className="opt-axis-select">
            <label>X:</label>
            <select value={xAxis} onChange={(e) => setXAxis(e.target.value)}>
              {targets.map((t) => (
                <option key={t} value={t}>
                  {TARGET_LABELS[t] || t}
                </option>
              ))}
            </select>
          </div>
          <div className="opt-axis-select">
            <label>Y:</label>
            <select value={yAxis} onChange={(e) => setYAxis(e.target.value)}>
              {targets.map((t) => (
                <option key={t} value={t}>
                  {TARGET_LABELS[t] || t}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <ChartNavigator chartId="pareto-front" minHeight={340}>
        <div className="opt-pareto-chart" style={{ width: "100%" }}>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
              />
              <XAxis
                type="number"
                dataKey="x"
                name={TARGET_LABELS[xAxis] || xAxis}
                stroke="var(--text-secondary)"
                tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                label={{
                  value: TARGET_LABELS[xAxis] || xAxis,
                  position: "insideBottomRight",
                  offset: -5,
                  fill: "var(--text-secondary)",
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={TARGET_LABELS[yAxis] || yAxis}
                stroke="var(--text-secondary)"
                tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                label={{
                  value: TARGET_LABELS[yAxis] || yAxis,
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                  fill: "var(--text-secondary)",
                  fontSize: 12,
                }}
              />
              <ZAxis range={[40, 120]} />
              <Tooltip
                content={({ payload }) => {
                  if (!payload || !payload[0]) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="opt-tooltip">
                      <div className="opt-tooltip-formula">{d.formula}</div>
                      <div className="opt-tooltip-tag">
                        <span
                          className="opt-tooltip-dot"
                          style={{ background: d.color }}
                        />
                        {d.tag}
                      </div>
                      {targets.map((t) => (
                        <div key={t} className="opt-tooltip-row">
                          <span>{TARGET_LABELS[t] || t}:</span>
                          <strong>{(d[t] ?? 0).toFixed(1)}</strong>
                        </div>
                      ))}
                      <div className="opt-tooltip-rank">Rank #{d.rank}</div>
                    </div>
                  );
                }}
              />
              <Scatter data={data}>
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </ChartNavigator>

      {/* Legend */}
      <div className="opt-legend">
        {uniqueTags.map((tag) => (
          <div key={tag} className="opt-legend-item">
            <span
              className="opt-legend-dot"
              style={{
                background:
                  USE_CASE_COLORS[tag] ||
                  solutions.find((s) => s.use_case_tag === tag)?.use_case_color ||
                  "#6B7280",
              }}
            />
            <span>{tag}</span>
          </div>
        ))}
      </div>

      <div className="opt-chart-summary">
        {solutions.length} Pareto-optimal solutions found
      </div>
    </div>
  );
}
