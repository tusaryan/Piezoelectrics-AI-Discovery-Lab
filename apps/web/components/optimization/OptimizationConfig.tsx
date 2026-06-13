"use client";

import { useEffect } from "react";
import { Target, Zap, Settings2 } from "lucide-react";
import { useOptimizationStore } from "@/lib/store/optimizationStore";
import type { ObjectiveConfig } from "@/lib/api/optimization";

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃ (pC/N)",
  tc: "Tc (°C)",
  vickers_hardness: "Hardness (HV)",
};

const TARGET_ICONS: Record<string, string> = {
  d33: "⚡",
  tc: "🌡️",
  vickers_hardness: "💎",
};

export default function OptimizationConfig() {
  const {
    presets,
    activePreset,
    objectives,
    popSize,
    nGenerations,
    optimizationLoading,
    loadPresets,
    setPreset,
    setObjective,
    setPopSize,
    setNGenerations,
    runOptimization,
  } = useOptimizationStore();

  useEffect(() => {
    loadPresets();
  }, [loadPresets]);

  const handleObjectiveChange = (
    target: string,
    field: keyof ObjectiveConfig,
    value: number | string,
  ) => {
    const current = objectives[target] || {
      direction: "maximize",
      min: 0,
      max: 1000,
      weight: 1.0,
    };
    setObjective(target, { ...current, [field]: value });
  };

  return (
    <div className="opt-card">
      <div className="opt-card-header">
        <Target size={18} />
        <h3>Optimization Config</h3>
      </div>
      <p className="opt-card-description">
        Define target property ranges and weights. Use presets for common use cases.
      </p>

      {/* Use-case presets */}
      <div className="opt-presets">
        <label className="opt-section-label">Use-Case Presets</label>
        <div className="opt-preset-grid">
          {presets.map((p) => (
            <button
              key={p.key}
              className={`opt-preset-btn ${activePreset === p.key ? "active" : ""}`}
              onClick={() => setPreset(p.key)}
            >
              <span className="opt-preset-label">{p.label}</span>
              <span className="opt-preset-desc">{p.description}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Objective configuration per target */}
      <div className="opt-objectives">
        <label className="opt-section-label">Target Objectives</label>
        {Object.entries(objectives).map(([target, config]) => (
          <div key={target} className="opt-objective-row">
            <div className="opt-objective-header">
              <span className="opt-objective-icon">
                {TARGET_ICONS[target] || "📊"}
              </span>
              <span className="opt-objective-name">
                {TARGET_LABELS[target] || target}
              </span>
              <select
                className="opt-direction-select"
                value={config.direction}
                onChange={(e) =>
                  handleObjectiveChange(target, "direction", e.target.value)
                }
              >
                <option value="maximize">Maximize ↑</option>
                <option value="minimize">Minimize ↓</option>
              </select>
            </div>
            <div className="opt-objective-controls">
              <div className="opt-range-input">
                <label>Min</label>
                <input
                  type="number"
                  value={config.min}
                  onChange={(e) =>
                    handleObjectiveChange(target, "min", Number(e.target.value))
                  }
                />
              </div>
              <div className="opt-range-input">
                <label>Max</label>
                <input
                  type="number"
                  value={config.max}
                  onChange={(e) =>
                    handleObjectiveChange(target, "max", Number(e.target.value))
                  }
                />
              </div>
              <div className="opt-range-input">
                <label>Weight</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={config.weight}
                  onChange={(e) =>
                    handleObjectiveChange(
                      target,
                      "weight",
                      Number(e.target.value),
                    )
                  }
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Algorithm parameters */}
      <div className="opt-params">
        <label className="opt-section-label">
          <Settings2 size={14} />
          Algorithm Parameters
        </label>
        <div className="opt-param-grid">
          <div className="opt-param">
            <label>Population Size</label>
            <input
              type="number"
              min={20}
              max={500}
              value={popSize}
              onChange={(e) => setPopSize(Number(e.target.value))}
            />
            <span className="opt-param-hint">
              More = better coverage, slower
            </span>
          </div>
          <div className="opt-param">
            <label>Generations</label>
            <input
              type="number"
              min={10}
              max={300}
              value={nGenerations}
              onChange={(e) => setNGenerations(Number(e.target.value))}
            />
            <span className="opt-param-hint">
              More = better convergence, slower
            </span>
          </div>
        </div>
      </div>

      {/* Run button */}
      <button
        className="opt-run-btn"
        onClick={() => runOptimization()}
        disabled={optimizationLoading}
      >
        {optimizationLoading ? (
          <>
            <span className="opt-spinner" />
            Running NSGA-II…
          </>
        ) : (
          <>
            <Zap size={16} />
            Run Optimization
          </>
        )}
      </button>
    </div>
  );
}
