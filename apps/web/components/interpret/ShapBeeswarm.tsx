"use client";

/**
 * SHAP Beeswarm Chart — D3-rendered global feature importance plot.
 *
 * Each dot = one sample. X-axis = SHAP value. Y-axis = feature (sorted by importance).
 * Color = feature value (low=blue, high=red).
 */

import { useEffect, useRef, useCallback, useState } from "react";
import { Loader2, ScatterChart, Maximize2, Minimize2 } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";
import InfoTooltip from "./InfoTooltip";
import { ChartNavigation } from "./ChartNavigation";

export default function ShapBeeswarm() {
  const { beeswarm, beeswarmLoading, beeswarmError, selectedModelId, fetchBeeswarm } =
    useInterpretStore();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (selectedModelId && !beeswarm && !beeswarmLoading) {
      fetchBeeswarm();
    }
  }, [selectedModelId, beeswarm, beeswarmLoading, fetchBeeswarm]);

  const drawChart = useCallback(() => {
    if (!beeswarm || !svgRef.current) return;

    const svg = svgRef.current;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const containerWidth = containerRef.current?.clientWidth ?? 700;
    const width = Math.min(containerWidth - 20, 900);
    const maxFeatures = 15;

    // Sort features by importance, take top N
    const indices = beeswarm.mean_abs_shap
      .map((v, i) => ({ v, i }))
      .sort((a, b) => b.v - a.v)
      .slice(0, maxFeatures)
      .map((x) => x.i);

    const featureNames = indices.map((i) => beeswarm.feature_names[i]);
    const height = featureNames.length * 28 + 80;

    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    const margin = { top: 30, right: 60, bottom: 40, left: 180 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    // Create group
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("transform", `translate(${margin.left},${margin.top})`);
    svg.appendChild(g);

    // Compute scales
    let shapMin = Infinity, shapMax = -Infinity;
    for (const fi of indices) {
      for (const row of beeswarm.shap_values) {
        const v = row[fi];
        if (v < shapMin) shapMin = v;
        if (v > shapMax) shapMax = v;
      }
    }
    const shapRange = Math.max(Math.abs(shapMin), Math.abs(shapMax)) * 1.1;

    // X scale
    const xScale = (v: number) =>
      ((v + shapRange) / (2 * shapRange)) * plotW;

    // Y scale — band scale for features
    const yBand = plotH / featureNames.length;
    const yScale = (idx: number) => idx * yBand + yBand / 2;

    // Color scale — blue (low) → red (high) via feature value
    const colorScale = (normalized: number) => {
      const r = Math.round(normalized * 255);
      const b = Math.round((1 - normalized) * 255);
      return `rgb(${r}, 60, ${b})`;
    };

    // Draw zero line
    const zeroLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    zeroLine.setAttribute("x1", String(xScale(0)));
    zeroLine.setAttribute("x2", String(xScale(0)));
    zeroLine.setAttribute("y1", "0");
    zeroLine.setAttribute("y2", String(plotH));
    zeroLine.setAttribute("stroke", "var(--border)");
    zeroLine.setAttribute("stroke-dasharray", "3,3");
    zeroLine.setAttribute("opacity", "0.5");
    g.appendChild(zeroLine);

    // Draw feature labels
    featureNames.forEach((name, idx) => {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", "-8");
      label.setAttribute("y", String(yScale(idx) + 4));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("fill", "var(--text-secondary)");
      label.setAttribute("font-size", "11");
      label.setAttribute("font-family", "var(--font-mono)");
      // Truncate long names
      const short = name.length > 22 ? name.slice(0, 20) + "…" : name;
      label.textContent = short;
      g.appendChild(label);
    });

    // Draw dots
    indices.forEach((fi, yIdx) => {
      // Get min/max for this feature's values (for color normalization)
      let fMin = Infinity, fMax = -Infinity;
      for (const row of beeswarm.feature_values) {
        const v = row[fi];
        if (v < fMin) fMin = v;
        if (v > fMax) fMax = v;
      }
      const fRange = fMax - fMin || 1;

      beeswarm.shap_values.forEach((shapRow, sIdx) => {
        const sx = shapRow[fi];
        const fv = beeswarm.feature_values[sIdx]?.[fi] ?? 0;
        const normalized = (fv - fMin) / fRange;

        // Add jitter
        const jitter = (Math.random() - 0.5) * yBand * 0.7;

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", String(xScale(sx)));
        circle.setAttribute("cy", String(yScale(yIdx) + jitter));
        circle.setAttribute("r", "2.5");
        circle.setAttribute("fill", colorScale(normalized));
        circle.setAttribute("opacity", "0.7");
        g.appendChild(circle);
      });
    });

    // X-axis label
    const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xLabel.setAttribute("x", String(plotW / 2));
    xLabel.setAttribute("y", String(plotH + 30));
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.setAttribute("fill", "var(--text-secondary)");
    xLabel.setAttribute("font-size", "12");
    xLabel.textContent = "SHAP value (impact on model output)";
    g.appendChild(xLabel);

    // Color legend
    const legendG = document.createElementNS("http://www.w3.org/2000/svg", "g");
    legendG.setAttribute("transform", `translate(${plotW + 10}, 0)`);

    const gradId = "shap-grad";
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const grad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
    grad.setAttribute("id", gradId);
    grad.setAttribute("x1", "0"); grad.setAttribute("y1", "1");
    grad.setAttribute("x2", "0"); grad.setAttribute("y2", "0");
    const s1 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
    s1.setAttribute("offset", "0%"); s1.setAttribute("stop-color", "rgb(0,60,255)");
    const s2 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
    s2.setAttribute("offset", "100%"); s2.setAttribute("stop-color", "rgb(255,60,0)");
    grad.appendChild(s1); grad.appendChild(s2);
    defs.appendChild(grad);
    svg.appendChild(defs);

    const lgH = Math.min(plotH * 0.5, 120);
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", "0"); rect.setAttribute("y", String((plotH - lgH) / 2));
    rect.setAttribute("width", "12"); rect.setAttribute("height", String(lgH));
    rect.setAttribute("rx", "2");
    rect.setAttribute("fill", `url(#${gradId})`);
    legendG.appendChild(rect);

    ["High", "Low"].forEach((label, i) => {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", "18");
      t.setAttribute("y", String((plotH - lgH) / 2 + (i === 0 ? 8 : lgH)));
      t.setAttribute("fill", "var(--text-muted)");
      t.setAttribute("font-size", "9");
      t.textContent = label;
      legendG.appendChild(t);
    });

    const ftLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    ftLabel.setAttribute("x", "6");
    ftLabel.setAttribute("y", String((plotH - lgH) / 2 - 8));
    ftLabel.setAttribute("fill", "var(--text-muted)");
    ftLabel.setAttribute("font-size", "9");
    ftLabel.setAttribute("text-anchor", "middle");
    ftLabel.textContent = "Feature";
    legendG.appendChild(ftLabel);

    const vlLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    vlLabel.setAttribute("x", "6");
    vlLabel.setAttribute("y", String((plotH - lgH) / 2 - 0));
    vlLabel.setAttribute("fill", "var(--text-muted)");
    vlLabel.setAttribute("font-size", "9");
    vlLabel.setAttribute("text-anchor", "middle");

    g.appendChild(legendG);
  }, [beeswarm]);

  useEffect(() => {
    drawChart();
    const handleResize = () => drawChart();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawChart]);

  return (
    <div className={`interpret-card ${expanded ? "expanded" : ""}`} id="shap-beeswarm">
      <div className="interpret-card-header">
        <div className="interpret-card-title">
          <ScatterChart size={16} />
          <span>SHAP Beeswarm — Global Feature Importance</span>
        </div>
        <div className="interpret-card-actions">
          <InfoTooltip title="SHAP Beeswarm Plot">
            <p>Each dot represents one sample from the training data.</p>
            <p><strong>X-axis:</strong> SHAP value — how much this feature pushed the prediction higher (right) or lower (left).</p>
            <p><strong>Color:</strong> Blue = low feature value, Red = high feature value.</p>
            <p><strong>Interpretation:</strong> Features at the top have the most impact on predictions overall.</p>
          </InfoTooltip>
          <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      <div className="interpret-card-body" ref={containerRef}>
        {beeswarmLoading && (
          <div className="interpret-loading">
            <Loader2 size={20} className="spin" />
            <span>Computing SHAP values...</span>
          </div>
        )}
        {beeswarmError && (
          <div className="interpret-error">{beeswarmError}</div>
        )}
        {beeswarm && !beeswarmLoading && (
          <div className="beeswarm-container" ref={containerRef}>
            <ChartNavigation containerRef={containerRef} id="shap-beeswarm">
              <svg ref={svgRef} />
            </ChartNavigation>
            <div className="beeswarm-meta">
              {beeswarm.n_samples} samples analyzed • {beeswarm.feature_names.length} features
            </div>
          </div>
        )}
        {!beeswarm && !beeswarmLoading && !beeswarmError && (
          <div className="interpret-empty">
            Select a model above to generate SHAP analysis
          </div>
        )}
      </div>
    </div>
  );
}
