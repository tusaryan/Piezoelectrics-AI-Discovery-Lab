"use client";

import {
  Database,
  BrainCircuit,
  Zap,
  BarChart3,
  HardDrive,
  Layers,
  TrendingUp,
  AlertCircle,
} from "lucide-react";
import type { SystemStats } from "@/lib/api/dashboard";

interface StatsCardsProps {
  stats: SystemStats | null;
  loading: boolean;
}

const STAT_ITEMS = [
  {
    key: "dataset_count" as const,
    label: "Datasets",
    icon: Database,
    color: "var(--color-primary)",
    sub: (s: SystemStats) => `${s.dataset_ready_count} ready · ${s.dataset_pending_count} pending`,
  },
  {
    key: "trained_model_count" as const,
    label: "Trained Models",
    icon: BrainCircuit,
    color: "#10B981",
    sub: (s: SystemStats) => `${s.total_material_rows.toLocaleString()} material rows`,
  },
  {
    key: "prediction_count" as const,
    label: "Predictions",
    icon: Zap,
    color: "#F59E0B",
    sub: (s: SystemStats) => `${s.training_completed_count} training jobs completed`,
  },
  {
    key: "db_size_mb" as const,
    label: "Database Size",
    icon: HardDrive,
    color: "#8B5CF6",
    sub: (s: SystemStats) => `${s.training_job_count} total jobs`,
    format: (v: number) => `${v.toFixed(1)} MB`,
  },
];

export default function StatsCards({ stats, loading }: StatsCardsProps) {
  return (
    <div className="stats-grid">
      {STAT_ITEMS.map((item) => {
        const Icon = item.icon;
        const value = stats ? stats[item.key] : 0;
        const formatted = item.format ? item.format(value as number) : String(value);

        return (
          <div key={item.key} className="stat-card">
            <div className="stat-card-header">
              <div className="stat-card-icon" style={{ color: item.color }}>
                <Icon size={20} />
              </div>
              <span className="stat-card-label">{item.label}</span>
            </div>
            <div className="stat-card-value">
              {loading ? (
                <span className="stat-skeleton" />
              ) : (
                formatted
              )}
            </div>
            {stats && (
              <div className="stat-card-sub">{item.sub(stats)}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
