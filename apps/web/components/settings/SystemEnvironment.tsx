"use client";

import { useEffect } from "react";
import { Server, Database, Brain, Cpu, HardDrive, BarChart3 } from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";

/**
 * SystemEnvironment — Shows system overview stats (dataset count, models, etc.)
 */
export default function SystemEnvironment() {
  const { systemEnv, systemLoading, fetchSystemEnv } = useSettingsStore();

  useEffect(() => { fetchSystemEnv(); }, [fetchSystemEnv]);

  const stats = [
    { icon: Database, label: "Datasets", value: systemEnv?.dataset_count ?? 0, sub: `${systemEnv?.total_rows ?? 0} total rows` },
    { icon: Brain, label: "Trained Models", value: systemEnv?.trained_model_count ?? 0, sub: "ML models" },
    { icon: BarChart3, label: "Predictions", value: systemEnv?.prediction_count ?? 0, sub: "total predictions" },
    { icon: HardDrive, label: "Database Size", value: `${systemEnv?.db_size_mb ?? 0} MB`, sub: "PostgreSQL" },
  ];

  const flags = [
    { label: "Composite Module", active: systemEnv?.enable_composite ?? false },
    { label: "Hardness Module", active: systemEnv?.enable_hardness ?? false },
    { label: "GNN Module", active: systemEnv?.enable_gnn ?? false },
  ];

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Server size={18} /></div>
        <div>
          <h3>System Environment</h3>
          <p className="settings-section-desc">Platform overview and runtime status</p>
        </div>
      </div>

      {systemLoading ? (
        <div className="settings-loading">Loading system info...</div>
      ) : (
        <>
          <div className="sys-stats-grid">
            {stats.map((s) => {
              const Icon = s.icon;
              return (
                <div key={s.label} className="sys-stat-card">
                  <div className="sys-stat-icon"><Icon size={16} /></div>
                  <div className="sys-stat-value">{s.value}</div>
                  <div className="sys-stat-label">{s.label}</div>
                  <div className="sys-stat-sub">{s.sub}</div>
                </div>
              );
            })}
          </div>

          <div className="sys-info-row">
            <div className="sys-info-item">
              <Cpu size={14} />
              <span>Python {systemEnv?.python_version || "—"}</span>
            </div>
            <div className="sys-info-item">
              <span className="sys-info-label">Version</span>
              <span className="sys-info-mono">v{systemEnv?.app_version || "2.1.0"}</span>
            </div>
          </div>

          <div className="sys-flags">
            {flags.map((f) => (
              <div key={f.label} className={`sys-flag ${f.active ? "active" : "inactive"}`}>
                <span className="sys-flag-dot" />
                {f.label}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
