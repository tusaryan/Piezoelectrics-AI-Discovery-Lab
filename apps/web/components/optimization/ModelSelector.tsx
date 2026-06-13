"use client";

import { useEffect } from "react";
import { FlaskConical, Star, Cpu } from "lucide-react";
import { useOptimizationStore } from "@/lib/store/optimizationStore";

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃ (pC/N)",
  tc: "Tc (°C)",
  vickers_hardness: "Hardness (HV)",
};

const TARGET_COLORS: Record<string, string> = {
  d33: "var(--chart-d33)",
  tc: "var(--chart-tc)",
  vickers_hardness: "var(--chart-hardness)",
};

export default function OptModelSelector() {
  const {
    models,
    modelsLoading,
    selectedModelIds,
    fetchModels,
    setSelectedModel,
  } = useOptimizationStore();

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Group models by target
  const byTarget: Record<string, typeof models> = {};
  for (const m of models) {
    if (!byTarget[m.target]) byTarget[m.target] = [];
    byTarget[m.target].push(m);
  }

  if (modelsLoading) {
    return (
      <div className="opt-card">
        <div className="opt-card-header">
          <Cpu size={18} />
          <h3>Select Models</h3>
        </div>
        <div className="opt-loading-skeleton">Loading models…</div>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="opt-card">
        <div className="opt-card-header">
          <Cpu size={18} />
          <h3>Select Models</h3>
        </div>
        <div className="opt-empty-state">
          <FlaskConical size={32} />
          <p>No trained models found. Train models first in the Train section.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="opt-card">
      <div className="opt-card-header">
        <Cpu size={18} />
        <h3>Surrogate Models</h3>
        <span className="opt-badge">{models.length} available</span>
      </div>
      <p className="opt-card-description">
        Select trained models as fitness functions for each optimization target.
      </p>
      <div className="opt-model-grid">
        {Object.entries(byTarget).map(([target, targetModels]) => (
          <div key={target} className="opt-model-group">
            <label
              className="opt-model-label"
              style={{ borderLeftColor: TARGET_COLORS[target] || "var(--primary)" }}
            >
              {TARGET_LABELS[target] || target}
            </label>
            <select
              className="opt-model-select"
              value={selectedModelIds[target] || ""}
              onChange={(e) => setSelectedModel(target, e.target.value)}
            >
              <option value="">— Skip —</option>
              {targetModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name} ({m.algorithm}, R²={m.r2_score.toFixed(3)})
                  {m.is_default ? " ★" : ""}
                </option>
              ))}
            </select>
            {selectedModelIds[target] && (
              <div className="opt-model-info">
                {(() => {
                  const sel = targetModels.find(
                    (m) => m.id === selectedModelIds[target],
                  );
                  if (!sel) return null;
                  return (
                    <>
                      <span className="opt-metric">
                        R² {sel.r2_score.toFixed(3)}
                      </span>
                      <span className="opt-metric">
                        RMSE {sel.rmse.toFixed(2)}
                      </span>
                      <span className="opt-metric">
                        {sel.n_train_samples} samples
                      </span>
                      {sel.is_default && (
                        <Star size={12} className="opt-default-star" />
                      )}
                    </>
                  );
                })()}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
