"use client";

/**
 * InterpretModelSelector — select a trained model + target for analysis.
 */

import { useEffect } from "react";
import { Cpu, Star, Loader2 } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";

export default function InterpretModelSelector() {
  const {
    models,
    selectedModelId,
    modelsLoading,
    loadModels,
    selectModel,
  } = useInterpretStore();

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const targetLabel = (t: string) => {
    if (t === "d33") return "d₃₃";
    if (t === "tc") return "Tc";
    if (t === "vickers_hardness") return "Hardness";
    return t;
  };

  return (
    <div className="interpret-model-selector">
      <div className="selector-header">
        <Cpu size={18} />
        <span>Select Model</span>
      </div>

      {modelsLoading ? (
        <div className="selector-loading">
          <Loader2 size={16} className="spin" />
          <span>Loading models...</span>
        </div>
      ) : models.length === 0 ? (
        <div className="selector-empty">
          No trained models available. Train a model first.
        </div>
      ) : (
        <div className="selector-grid">
          {models.map((m) => (
            <button
              key={m.id}
              className={`model-card-btn ${selectedModelId === m.id ? "active" : ""}`}
              onClick={() => selectModel(m.id)}
            >
              <div className="model-card-top">
                <span className={`target-badge target-${m.target}`}>
                  {targetLabel(m.target)}
                </span>
                {m.is_default && (
                  <Star size={12} className="default-star" />
                )}
              </div>
              <div className="model-card-name">{m.display_name}</div>
              <div className="model-card-meta">
                <span>{m.algorithm}</span>
                <span>R² {(m.r2_score ?? 0).toFixed(3)}</span>
                <span>{m.n_train_samples} samples</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
