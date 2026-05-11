"use client";

/**
 * SHAP Dependence Plot — scatter plot of feature value vs SHAP value.
 *
 * Reveals non-linear relationships between a feature and model predictions.
 * Color encodes auto-detected interaction feature.
 */

import { useEffect, useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Loader2, TrendingDown, Maximize2, Minimize2 } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";
import InfoTooltip from "./InfoTooltip";

export default function ShapDependence() {
  const {
    dependence, dependenceLoading, dependenceError,
    selectedModelId, beeswarm, fetchDependence, dependenceFeature,
  } = useInterpretStore();
  const [expanded, setExpanded] = useState(false);

  // Feature dropdown
  const featureOptions = beeswarm?.feature_names ?? [];
  const [selectedFeature, setSelectedFeature] = useState<string | null>(dependenceFeature);

  useEffect(() => {
    if (beeswarm && !selectedFeature && beeswarm.feature_names.length > 0) {
      // Auto-select most important feature and fetch
      const topIdx = beeswarm.mean_abs_shap.indexOf(Math.max(...beeswarm.mean_abs_shap));
      const topFeat = beeswarm.feature_names[topIdx];
      setSelectedFeature(topFeat);
      fetchDependence(topFeat);
    }
  }, [beeswarm, selectedFeature, fetchDependence]);

  const handleFeatureChange = (feat: string) => {
    setSelectedFeature(feat);
    fetchDependence(feat);
  };

  // Build scatter data
  const scatterData = dependence
    ? dependence.feature_values.map((fv, i) => ({
        x: fv,
        y: dependence.shap_values[i],
        interaction: dependence.interaction_values[i] ?? 0,
      }))
    : [];

  // Normalize interaction for color
  const intMin = scatterData.length > 0 ? Math.min(...scatterData.map(d => d.interaction)) : 0;
  const intMax = scatterData.length > 0 ? Math.max(...scatterData.map(d => d.interaction)) : 1;
  const intRange = intMax - intMin || 1;

  const getColor = (val: number) => {
    const n = (val - intMin) / intRange;
    const r = Math.round(n * 230 + 25);
    const b = Math.round((1 - n) * 230 + 25);
    return `rgb(${r}, 80, ${b})`;
  };

  return (
    <div className={`interpret-card ${expanded ? "expanded" : ""}`} id="shap-dependence">
      <div className="interpret-card-header">
        <div className="interpret-card-title">
          <TrendingDown size={16} />
          <span>Feature Dependence Plot</span>
        </div>
        <div className="interpret-card-actions">
          {featureOptions.length > 0 && (
            <select
              className="dependence-feature-select"
              value={selectedFeature ?? ""}
              onChange={(e) => handleFeatureChange(e.target.value)}
            >
              <option value="" disabled>Select feature</option>
              {featureOptions.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          )}
          <InfoTooltip title="Feature Dependence Plot">
            <p>Shows the relationship between a specific feature&apos;s value and its SHAP value.</p>
            <p><strong>X-axis:</strong> Feature value in training data.</p>
            <p><strong>Y-axis:</strong> SHAP value — contribution to prediction.</p>
            <p><strong>Color:</strong> Auto-detected interaction feature — reveals which other feature modifies this relationship.</p>
            <p>Non-linear patterns indicate complex learned relationships.</p>
          </InfoTooltip>
          <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      <div className="interpret-card-body">
        {dependenceLoading && (
          <div className="interpret-loading">
            <Loader2 size={20} className="spin" />
            <span>Computing dependence...</span>
          </div>
        )}
        {dependenceError && <div className="interpret-error">{dependenceError}</div>}
        {dependence && !dependenceLoading && scatterData.length > 0 && (
          <div className="dependence-chart-wrapper">
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 10, right: 30, bottom: 40, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                <XAxis
                  dataKey="x" type="number" name={dependence.feature_name}
                  tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                  label={{
                    value: dependence.feature_name, position: "bottom", offset: 20,
                    fill: "var(--text-secondary)", fontSize: 12,
                  }}
                />
                <YAxis
                  dataKey="y" type="number" name="SHAP value"
                  tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                  label={{
                    value: "SHAP value", angle: -90, position: "insideLeft", offset: -5,
                    fill: "var(--text-secondary)", fontSize: 12,
                  }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    background: "var(--card)", border: "1px solid var(--border)",
                    borderRadius: "8px", fontSize: "12px", color: "var(--text)",
                  }}
                  formatter={(value: number, name: string) => [value.toFixed(4), name]}
                />
                <Scatter data={scatterData}>
                  {scatterData.map((entry, index) => (
                    <Cell key={index} fill={getColor(entry.interaction)} opacity={0.7} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            {dependence.interaction_feature && (
              <div className="dependence-interaction-label">
                Color: <span className="font-mono">{dependence.interaction_feature}</span>
              </div>
            )}
          </div>
        )}
        {!dependence && !dependenceLoading && !dependenceError && (
          <div className="interpret-empty">
            {selectedFeature ? "Click a feature to analyze" : "Select a feature from the dropdown"}
          </div>
        )}
      </div>
    </div>
  );
}
