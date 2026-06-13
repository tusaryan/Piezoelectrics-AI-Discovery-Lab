"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { TrendingUp } from "lucide-react";
import { useOptimizationStore } from "@/lib/store/optimizationStore";
import ChartNavigator from "@/components/ui/ChartNavigator";

const TARGET_COLORS: Record<string, string> = {
  d33: "#6366F1",
  tc: "#10B981",
  vickers_hardness: "#F59E0B",
};

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃",
  tc: "Tc",
  vickers_hardness: "Hardness",
};

export default function OptConvergenceChart() {
  const { convergence, optimizationStats } = useOptimizationStore();
  const targets = optimizationStats?.targets_optimized || [];

  if (convergence.length === 0) return null;

  return (
    <div className="opt-card">
      <div className="opt-card-header">
        <TrendingUp size={18} />
        <h3>Convergence</h3>
        {optimizationStats && (
          <span className="opt-badge">
            {optimizationStats.n_generations_run} gen •{" "}
            {optimizationStats.duration_seconds.toFixed(1)}s
          </span>
        )}
      </div>
      <p className="opt-card-description">
        Average objective values across generations — lower trend indicates convergence.
      </p>

      <ChartNavigator chartId="convergence" minHeight={260}>
        <div className="opt-convergence-chart" style={{ width: "100%" }}>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart
              data={convergence}
              margin={{ top: 10, right: 20, bottom: 5, left: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
              />
              <XAxis
                dataKey="generation"
                stroke="var(--text-secondary)"
                tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                label={{
                  value: "Generation",
                  position: "insideBottomRight",
                  offset: -5,
                  fill: "var(--text-secondary)",
                  fontSize: 11,
                }}
              />
              <YAxis
                stroke="var(--text-secondary)"
                tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--card)",
                  border: "1px solid var(--border)",
                  borderRadius: "8px",
                  color: "var(--text)",
                  fontSize: "12px",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", color: "var(--text-secondary)" }}
              />
              {targets.map((t) => (
                <Line
                  key={t}
                  type="monotone"
                  dataKey={t}
                  name={TARGET_LABELS[t] || t}
                  stroke={TARGET_COLORS[t] || "#6366F1"}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartNavigator>

      {optimizationStats && (
        <div className="opt-stats-row">
          <div className="opt-stat">
            <span className="opt-stat-label">Evaluations</span>
            <span className="opt-stat-value">
              {optimizationStats.n_evaluations.toLocaleString()}
            </span>
          </div>
          <div className="opt-stat">
            <span className="opt-stat-label">Duration</span>
            <span className="opt-stat-value">
              {optimizationStats.duration_seconds.toFixed(1)}s
            </span>
          </div>
          <div className="opt-stat">
            <span className="opt-stat-label">Targets</span>
            <span className="opt-stat-value">
              {optimizationStats.targets_optimized
                .map((t) => TARGET_LABELS[t] || t)
                .join(", ")}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
