"use client";

/**
 * ModelSelector — Per-target model dropdowns for d33, tc, hardness.
 * User picks one model per property (or "Skip" to not predict that property).
 */

import { useEffect, useState, useRef } from "react";
import { Brain, ChevronDown, Star, Info } from "lucide-react";
import { listModels } from "@/lib/api/predictions";
import type { TrainedModelItem } from "@/lib/api/predictions";
import { usePredictStore } from "@/lib/store/predictStore";

const TARGETS = [
  { key: "d33" as const, label: "d₃₃ (Piezoelectric Coefficient)", color: "var(--chart-d33)" },
  { key: "tc" as const, label: "Tc (Curie Temperature)", color: "var(--chart-tc)" },
  { key: "vickers_hardness" as const, label: "Vickers Hardness", color: "var(--chart-hardness)" },
];

function TargetDropdown({
  target,
  models,
  selectedId,
  onSelect,
}: {
  target: typeof TARGETS[number];
  models: TrainedModelItem[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const targetModels = models.filter((m) => m.target === target.key);
  const selected = targetModels.find((m) => m.id === selectedId);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div className="target-model-row" ref={ref}>
      <div className="target-model-label">
        <span className="target-dot" style={{ background: target.color }} />
        <span className="target-name">{target.label}</span>
      </div>
      <div className="target-dropdown-wrapper">
        <button
          className={`target-dropdown-btn ${open ? "open" : ""} ${selected ? "has-value" : ""}`}
          onClick={() => setOpen(!open)}
          type="button"
        >
          <span className="target-dropdown-text">
            {selected ? (
              <>
                {selected.display_name}
                <span className="target-dropdown-meta">
                  R²={selected.r2_score.toFixed(3)}
                </span>
              </>
            ) : targetModels.length === 0 ? (
              <span className="target-dropdown-placeholder">No models trained</span>
            ) : (
              <span className="target-dropdown-placeholder">Skip — don&apos;t predict</span>
            )}
          </span>
          <ChevronDown size={14} className={`target-chevron ${open ? "rotated" : ""}`} />
        </button>

        {open && targetModels.length > 0 && (
          <div className="target-dropdown-menu">
            <button
              className={`target-dropdown-item ${!selectedId ? "active" : ""}`}
              onClick={() => { onSelect(null); setOpen(false); }}
            >
              <span className="target-dropdown-item-name">Skip — don&apos;t predict</span>
            </button>
            {targetModels.map((m) => (
              <button
                key={m.id}
                className={`target-dropdown-item ${selectedId === m.id ? "active" : ""}`}
                onClick={() => { onSelect(m.id); setOpen(false); }}
              >
                <span className="target-dropdown-item-name">
                  {m.display_name}
                  {m.is_default && <Star size={10} fill="currentColor" style={{ color: "var(--warning)", marginLeft: 4 }} />}
                </span>
                <span className="target-dropdown-item-meta">
                  R²={m.r2_score.toFixed(3)} · RMSE={m.rmse.toFixed(2)} · {m.n_train_samples} samples
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ModelSelector() {
  const {
    models,
    setModels,
    targetModels,
    setTargetModel,
    modelsLoading,
    setModelsLoading,
  } = usePredictStore();

  useEffect(() => {
    let cancelled = false;
    setModelsLoading(true);
    listModels()
      .then((data) => {
        if (cancelled) return;
        setModels(data);
        // Auto-select default models per target
        for (const t of TARGETS) {
          const defaultModel = data.find((m) => m.target === t.key && m.is_default);
          const anyModel = data.find((m) => m.target === t.key);
          if (defaultModel) setTargetModel(t.key, defaultModel.id);
          else if (anyModel) setTargetModel(t.key, anyModel.id);
        }
      })
      .catch(() => { if (!cancelled) setModels([]); })
      .finally(() => { if (!cancelled) setModelsLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (modelsLoading) {
    return (
      <div className="predict-card">
        <div className="predict-card-title"><Brain size={16} /> Select Models</div>
        <div className="predict-empty-state" style={{ padding: "20px 0" }}>
          <p>Loading trained models...</p>
        </div>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="predict-card">
        <div className="predict-card-title"><Brain size={16} /> Select Models</div>
        <div className="no-models-notice">
          <h3>No Trained Models</h3>
          <p>Train a model first in the Train section, then return here to predict.</p>
        </div>
      </div>
    );
  }

  const selectedCount = Object.values(targetModels).filter(Boolean).length;

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <Brain size={16} /> Select Models
        {selectedCount > 0 && (
          <span className="model-count-badge">{selectedCount} selected</span>
        )}
      </div>
      <div className="target-model-hint">
        <Info size={12} />
        Choose a model for each property you want to predict, or skip to exclude it.
      </div>
      <div className="target-model-grid">
        {TARGETS.map((t) => (
          <TargetDropdown
            key={t.key}
            target={t}
            models={models}
            selectedId={targetModels[t.key]}
            onSelect={(id) => setTargetModel(t.key, id)}
          />
        ))}
      </div>
    </div>
  );
}
