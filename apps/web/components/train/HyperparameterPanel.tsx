/**
 * HyperparameterPanel — dynamic per-algorithm parameter controls with "i" tooltips.
 */

"use client";

import { Info, RotateCcw } from "lucide-react";
import { useTrainingStore } from "@/lib/store/trainingStore";
import type { AlgorithmInfo } from "@/lib/api/training";
import { useState } from "react";

export default function HyperparameterPanel() {
  const {
    targets, algorithms, hyperparameters, algorithmList, mode,
    setHyperparam,
  } = useTrainingStore();

  if (mode === "auto") {
    return (
      <div className="hyperparam-panel">
        <div className="hyperparam-auto-notice">
          <Info size={16} />
          <p>
            Auto-tune mode is active. Optuna will automatically find optimal
            hyperparameters using Bayesian optimization with cross-validation.
          </p>
        </div>
      </div>
    );
  }

  // Get unique algorithms being used
  const activeAlgos = [...new Set(Object.values(algorithms))];

  return (
    <div className="hyperparam-panel">
      {activeAlgos.map((algoKey) => {
        const meta = algorithmList.find((a) => a.key === algoKey);
        if (!meta) return null;
        const relatedTargets = targets.filter((t) => algorithms[t] === algoKey);

        return (
          <AlgoParams
            key={algoKey}
            meta={meta}
            targets={relatedTargets}
            hyperparameters={hyperparameters}
            setHyperparam={setHyperparam}
          />
        );
      })}
    </div>
  );
}

function AlgoParams({
  meta,
  targets,
  hyperparameters,
  setHyperparam,
}: {
  meta: AlgorithmInfo;
  targets: string[];
  hyperparameters: Record<string, Record<string, number | string>>;
  setHyperparam: (target: string, param: string, value: number | string) => void;
}) {
  const [tooltipOpen, setTooltipOpen] = useState<string | null>(null);

  // Use the first target's params as representative (unified mode)
  const targetKey = targets[0] || "";
  const currentParams = hyperparameters[targetKey] || {};

  const handleChange = (name: string, value: number | string) => {
    for (const t of targets) {
      setHyperparam(t, name, value);
    }
  };

  const handleReset = () => {
    for (const [name, def] of Object.entries(meta.hyperparameters)) {
      for (const t of targets) {
        setHyperparam(t, name, def.default);
      }
    }
  };

  return (
    <div className="hyperparam-algo-section">
      <div className="hyperparam-algo-header">
        <h4>{meta.display_name}</h4>
        <span className="hyperparam-targets">
          {targets.join(", ")}
        </span>
        <button className="hyperparam-reset" onClick={handleReset} title="Reset to defaults">
          <RotateCcw size={14} />
        </button>
      </div>

      <div className="hyperparam-grid">
        {Object.entries(meta.hyperparameters).map(([name, def]) => {
          const val = currentParams[name] ?? def.default;

          return (
            <div key={name} className="hyperparam-row">
              <div className="hyperparam-label">
                <span>{name}</span>
                <button
                  className="hyperparam-info-btn"
                  onClick={() => setTooltipOpen(tooltipOpen === name ? null : name)}
                  aria-label={`Info about ${name}`}
                >
                  <Info size={13} />
                </button>
              </div>

              {tooltipOpen === name && (
                <div className="hyperparam-tooltip">
                  <p><strong>Description:</strong> {def.description}</p>
                  <p><strong>Impact:</strong> {def.impact}</p>
                  <p><strong>Recommended:</strong> {String(def.recommended)}</p>
                </div>
              )}

              {def.type === "select" && def.options ? (
                <select
                  className="hyperparam-select"
                  value={String(val)}
                  onChange={(e) => handleChange(name, e.target.value)}
                >
                  {def.options.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : (
                <div className="hyperparam-slider-row">
                  <input
                    type="range"
                    className="hyperparam-slider"
                    min={def.min ?? 0}
                    max={def.max ?? 100}
                    step={def.step ?? 1}
                    value={Number(val)}
                    onChange={(e) => handleChange(name, Number(e.target.value))}
                  />
                  <input
                    type="number"
                    className="hyperparam-number"
                    min={def.min ?? undefined}
                    max={def.max ?? undefined}
                    step={def.step ?? undefined}
                    value={Number(val)}
                    onChange={(e) => {
                      let v = Number(e.target.value);
                      if (def.min != null) v = Math.max(def.min, v);
                      if (def.max != null) v = Math.min(def.max, v);
                      handleChange(name, v);
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
