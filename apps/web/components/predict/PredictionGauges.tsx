"use client";

/**
 * PredictionGauges — Animated gauge bars for d33, tc, hardness with CI.
 */

import { Gauge } from "lucide-react";
import { usePredictStore } from "@/lib/store/predictStore";

interface GaugeBarProps {
  label: string;
  unit: string;
  value: number | null | undefined;
  ciLower: number | null | undefined;
  ciUpper: number | null | undefined;
  min: number;
  max: number;
  colorClass: string;
}

function GaugeBar({
  label,
  unit,
  value,
  ciLower,
  ciUpper,
  min,
  max,
  colorClass,
}: GaugeBarProps) {
  if (value == null) return null;

  const pct = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));

  return (
    <div className="gauge-item">
      <div className="gauge-header">
        <span className="gauge-label">{label}</span>
        <span className="gauge-value">
          {value.toFixed(1)}
          <span className="gauge-unit">{unit}</span>
        </span>
      </div>
      <div className="gauge-bar-track">
        <div
          className={`gauge-bar-fill ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {ciLower != null && ciUpper != null && (
        <div className="gauge-ci">
          95% CI:{" "}
          <span className="gauge-ci-range">
            [{ciLower.toFixed(1)} – {ciUpper.toFixed(1)}]
          </span>
        </div>
      )}
    </div>
  );
}

export default function PredictionGauges() {
  const { prediction } = usePredictStore();

  if (!prediction || prediction.status !== "success") return null;

  const hasAny =
    prediction.d33?.value != null ||
    prediction.tc?.value != null ||
    prediction.hardness?.value != null;

  if (!hasAny) return null;

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <Gauge size={16} /> Predicted Properties
      </div>
      <div className="gauges-grid">
        <GaugeBar
          label="d₃₃ (Piezoelectric Coefficient)"
          unit="pC/N"
          value={prediction.d33?.value}
          ciLower={prediction.d33?.ci_lower}
          ciUpper={prediction.d33?.ci_upper}
          min={0}
          max={700}
          colorClass="d33"
        />
        <GaugeBar
          label="Tc (Curie Temperature)"
          unit="°C"
          value={prediction.tc?.value}
          ciLower={prediction.tc?.ci_lower}
          ciUpper={prediction.tc?.ci_upper}
          min={0}
          max={500}
          colorClass="tc"
        />
        <GaugeBar
          label="Vickers Hardness"
          unit="HV"
          value={prediction.hardness?.value}
          ciLower={prediction.hardness?.ci_lower}
          ciUpper={prediction.hardness?.ci_upper}
          min={0}
          max={1200}
          colorClass="hardness"
        />
      </div>
    </div>
  );
}
