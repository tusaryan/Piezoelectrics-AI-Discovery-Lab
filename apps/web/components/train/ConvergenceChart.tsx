/**
 * ConvergenceChart — separate per-target convergence charts with independent Y-axes.
 * Each target gets its own chart so different metric scales don't flatten each other.
 */

"use client";

import { Info } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useTrainingStore } from "@/lib/store/trainingStore";
import { useState, useRef } from "react";
import { ChartNavigation } from "@/components/interpret/ChartNavigation";

const TARGET_COLORS: Record<string, string> = {
  d33: "hsl(245, 80%, 65%)",       // Indigo
  tc: "hsl(160, 70%, 45%)",        // Emerald
  vickers_hardness: "hsl(35, 90%, 55%)", // Amber
};

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃ (pC/N)",
  tc: "Tc (°C)",
  vickers_hardness: "Hardness (HV)",
};

export default function ConvergenceChart() {
  const convergenceData = useTrainingStore((s) => s.convergenceData);
  const jobPhase = useTrainingStore((s) => s.jobPhase);
  const [showGuide, setShowGuide] = useState(false);
  const gridRef = useRef<HTMLDivElement>(null);

  const targets = Object.keys(convergenceData).filter(
    (t) => convergenceData[t]?.length > 0
  );

  if (targets.length === 0) {
    if (jobPhase === "training") {
      return (
        <div className="convergence-chart empty">
          <p>Convergence data will appear here during training...</p>
        </div>
      );
    }
    return null;
  }

  return (
    <div className="convergence-chart">
      <div className="convergence-header">
        <h4>Convergence</h4>
        <button
          className="convergence-info-btn"
          onClick={() => setShowGuide(!showGuide)}
        >
          <Info size={14} />
        </button>
      </div>

      {showGuide && (
        <div className="convergence-guide">
          <p><strong>Good convergence:</strong> Smooth decrease that plateaus → model is learning and stabilizing.</p>
          <p><strong>Bad convergence:</strong> Erratic jumps or no decrease → try lower learning rate or more data.</p>
          <p><strong>Overfitting:</strong> Train metric keeps improving but validation doesn&apos;t → reduce complexity.</p>
        </div>
      )}

      {/* Render one chart per target — each has its own Y-axis scale */}
      <div className="convergence-charts-grid">
        <div ref={gridRef} style={{ position: "relative" }}>
          <ChartNavigation containerRef={gridRef} id="convergence">
        {targets.map((target) => {
          const points = convergenceData[target] || [];
          const chartData = points.map((p, i) => ({
            iteration: p.iteration ?? i,
            metric: p.metric,
          }));

          const color = TARGET_COLORS[target] || "hsl(200, 60%, 50%)";
          const label = TARGET_LABELS[target] || target;

          return (
            <div key={target} className="convergence-single-chart">
              <div className="convergence-chart-label">
                <span
                  className="convergence-dot"
                  style={{ background: color }}
                />
                <span className="convergence-target-name">{label}</span>
                <span className="convergence-point-count">
                  {points.length} iterations
                </span>
              </div>

              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                  <XAxis
                    dataKey="iteration"
                    stroke="var(--text-secondary)"
                    fontSize={10}
                    tick={{ fill: "var(--text-secondary)" }}
                  />
                  <YAxis
                    stroke="var(--text-secondary)"
                    fontSize={10}
                    tick={{ fill: "var(--text-secondary)" }}
                    domain={["auto", "auto"]}
                    tickFormatter={(v: number) =>
                      v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(1)
                    }
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-glass)",
                      border: "1px solid var(--border)",
                      borderRadius: "8px",
                      color: "var(--text-primary)",
                      fontSize: 12,
                    }}
                    formatter={(value: number) => [value.toFixed(4), target]}
                    labelFormatter={(label: number) => `Iteration ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="metric"
                    stroke={color}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                    name={target}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
        </ChartNavigation>
        </div>
      </div>
    </div>
  );
}
