"use client";

/**
 * SHAP Waterfall Chart — local prediction explanation for a single sample.
 *
 * Shows contribution of each feature (positive = red, negative = blue)
 * from base value to final prediction.
 */

import { useEffect, useState } from "react";
import { Loader2, Waves, Maximize2, Minimize2, ChevronLeft, ChevronRight } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";
import InfoTooltip from "./InfoTooltip";

export default function ShapWaterfall() {
  const {
    waterfall, waterfallLoading, waterfallError,
    selectedModelId, fetchWaterfall, waterfallSampleIndex,
  } = useInterpretStore();
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (selectedModelId && !waterfall && !waterfallLoading) {
      fetchWaterfall(0);
    }
  }, [selectedModelId, waterfall, waterfallLoading, fetchWaterfall]);

  const navigateSample = (delta: number) => {
    if (!waterfall) return;
    const next = Math.max(0, Math.min(waterfallSampleIndex + delta, waterfall.n_total_samples - 1));
    fetchWaterfall(next);
  };

  // Sort features by absolute SHAP contribution
  const getSortedFeatures = () => {
    if (!waterfall) return [];
    const items = waterfall.feature_names.map((name, i) => ({
      name,
      shap: waterfall.shap_values[i],
      value: waterfall.feature_values[i],
    }));
    items.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    return items.slice(0, 12); // Top 12
  };

  const formatFeatureName = (name: string) => {
    if (name.length > 28) return name.slice(0, 26) + "…";
    return name;
  };

  const sorted = getSortedFeatures();
  const maxAbsShap = sorted.length > 0 ? Math.max(...sorted.map(s => Math.abs(s.shap))) : 1;

  return (
    <div className={`interpret-card ${expanded ? "expanded" : ""}`} id="shap-waterfall">
      <div className="interpret-card-header">
        <div className="interpret-card-title">
          <Waves size={16} />
          <span>SHAP Waterfall — Local Explanation</span>
        </div>
        <div className="interpret-card-actions">
          {waterfall && (
            <div className="sample-nav">
              <button onClick={() => navigateSample(-1)} disabled={waterfallSampleIndex <= 0}>
                <ChevronLeft size={14} />
              </button>
              <span className="sample-label">
                Sample {waterfallSampleIndex + 1}/{waterfall.n_total_samples}
              </span>
              <button
                onClick={() => navigateSample(1)}
                disabled={waterfallSampleIndex >= waterfall.n_total_samples - 1}
              >
                <ChevronRight size={14} />
              </button>
            </div>
          )}
          <InfoTooltip title="SHAP Waterfall Plot">
            <p>Shows how each feature contributes to one specific prediction.</p>
            <p><strong>Red bars:</strong> features that increased the prediction.</p>
            <p><strong>Blue bars:</strong> features that decreased the prediction.</p>
            <p>Starts from base value (average prediction) and shows the push/pull of each feature.</p>
          </InfoTooltip>
          <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      <div className="interpret-card-body">
        {waterfallLoading && (
          <div className="interpret-loading">
            <Loader2 size={20} className="spin" />
            <span>Computing waterfall...</span>
          </div>
        )}
        {waterfallError && <div className="interpret-error">{waterfallError}</div>}
        {waterfall && !waterfallLoading && (
          <div className="waterfall-container">
            <div className="waterfall-summary">
              <span>Base: <strong>{waterfall.base_value.toFixed(2)}</strong></span>
              <span>→ Prediction: <strong>{waterfall.prediction.toFixed(2)}</strong></span>
            </div>
            <div className="waterfall-bars">
              {sorted.map((item) => {
                const pct = (Math.abs(item.shap) / maxAbsShap) * 100;
                const isPositive = item.shap > 0;
                return (
                  <div key={item.name} className="waterfall-bar-row">
                    <div className="waterfall-feature-name font-mono">
                      {formatFeatureName(item.name)}
                    </div>
                    <div className="waterfall-bar-track">
                      <div
                        className={`waterfall-bar ${isPositive ? "positive" : "negative"}`}
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      />
                    </div>
                    <div className={`waterfall-value ${isPositive ? "positive" : "negative"}`}>
                      {isPositive ? "+" : ""}{item.shap.toFixed(4)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {!waterfall && !waterfallLoading && !waterfallError && (
          <div className="interpret-empty">
            Select a model to see feature contributions
          </div>
        )}
      </div>
    </div>
  );
}
