"use client";

import { Star } from "lucide-react";
import type { DashboardModel } from "@/lib/api/dashboard";
import { useDashboardStore } from "@/lib/store/dashboardStore";

interface DefaultModelSelectorProps {
  models: DashboardModel[];
}

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃ (pC/N)",
  tc: "Tc (°C)",
  vickers_hardness: "Hardness (HV)",
};

const TARGETS = ["d33", "tc", "vickers_hardness"] as const;

export default function DefaultModelSelector({ models }: DefaultModelSelectorProps) {
  const { setDefaultModel } = useDashboardStore();

  // Group models by target
  const byTarget: Record<string, DashboardModel[]> = {};
  for (const m of models) {
    if (!byTarget[m.target]) byTarget[m.target] = [];
    byTarget[m.target].push(m);
  }

  return (
    <div className="dashboard-section default-selector">
      <h2 className="section-title">
        <Star size={18} /> Default Model Selector
      </h2>
      <p className="section-description">
        Choose which model is used for predictions by default for each target property.
      </p>

      <div className="default-selector-grid">
        {TARGETS.map((target) => {
          const targetModels = byTarget[target] || [];
          const current = targetModels.find((m) => m.is_default);

          return (
            <div key={target} className="default-target-card">
              <div className="default-target-label">
                {TARGET_LABELS[target] || target}
              </div>

              {targetModels.length === 0 ? (
                <div className="default-none">No models trained</div>
              ) : (
                <select
                  className="default-select"
                  value={current?.id || ""}
                  onChange={async (e) => {
                    if (e.target.value) {
                      await setDefaultModel(e.target.value);
                    }
                  }}
                >
                  <option value="">Select default…</option>
                  {targetModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.display_name} (R²={m.r2_score.toFixed(3)})
                    </option>
                  ))}
                </select>
              )}

              {current && (
                <div className="default-current-info">
                  <span className="default-algo">{current.algorithm}</span>
                  <span className="default-r2">R²={current.r2_score.toFixed(4)}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
