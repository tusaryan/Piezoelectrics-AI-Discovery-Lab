"use client";

import { PieChart } from "lucide-react";
import {
  PieChart as RechartsPie,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TargetDistribution } from "@/lib/api/dashboard";
import ChartNavigator from "@/components/ui/ChartNavigator";

interface TargetDistributionChartProps {
  data: TargetDistribution[];
}

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃",
  tc: "Tc",
  vickers_hardness: "Hardness",
};

const TARGET_COLORS: Record<string, string> = {
  d33: "#4F46E5",
  tc: "#10B981",
  vickers_hardness: "#F59E0B",
};

export default function TargetDistributionChart({ data }: TargetDistributionChartProps) {
  if (data.length === 0) {
    return (
      <div className="dashboard-section target-dist">
        <h2 className="section-title">
          <PieChart size={18} /> Target Distribution
        </h2>
        <div className="empty-state">
          <PieChart size={32} />
          <p>No models trained yet</p>
        </div>
      </div>
    );
  }

  const chartData = data.map((d) => ({
    name: TARGET_LABELS[d.target] || d.target,
    value: d.count,
    percentage: d.percentage,
    fill: TARGET_COLORS[d.target] || "#64748B",
  }));

  return (
    <div className="dashboard-section target-dist">
      <h2 className="section-title">
        <PieChart size={18} /> Target Distribution
      </h2>
      <div className="donut-container">
        <ChartNavigator chartId="target-distribution" minHeight={280}>
          <div style={{ width: "100%", position: "relative" }}>
            <ResponsiveContainer width="100%" height={280}>
              <RechartsPie>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={4}
                  dataKey="value"
                  stroke="none"
                  label={({ name, percentage }) => `${name} ${percentage}%`}
                  labelLine={false}
                >
                  {chartData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                    color: "var(--text)",
                  }}
                  formatter={(value: number, name: string) => [
                    `${value} model${value !== 1 ? "s" : ""}`,
                    name,
                  ]}
                />
                <Legend
                  wrapperStyle={{ fontSize: "12px", color: "var(--text-muted)" }}
                />
              </RechartsPie>
            </ResponsiveContainer>
            {/* Center label */}
            <div className="donut-center">
              <span className="donut-total">{data.reduce((a, b) => a + b.count, 0)}</span>
              <span className="donut-label">Models</span>
            </div>
          </div>
        </ChartNavigator>
      </div>
    </div>
  );
}