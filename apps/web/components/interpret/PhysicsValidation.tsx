"use client";

/**
 * Physics Validation Card — checks SHAP associations vs piezoelectric physics.
 */

import { useEffect, useState } from "react";
import { ShieldCheck, CheckCircle2, AlertTriangle, ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";
import InfoTooltip from "./InfoTooltip";

export default function PhysicsValidation() {
  const { physics, physicsLoading, physicsError, selectedModelId, fetchPhysicsValidation } =
    useInterpretStore();
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    if (selectedModelId && !physics && !physicsLoading) {
      fetchPhysicsValidation();
    }
  }, [selectedModelId, physics, physicsLoading, fetchPhysicsValidation]);

  const scoreColor = (score: number) => {
    if (score >= 80) return "var(--success)";
    if (score >= 50) return "var(--warning)";
    return "var(--error)";
  };

  const scoreLabel = (score: number) => {
    if (score >= 80) return "Strong Alignment";
    if (score >= 50) return "Moderate Alignment";
    return "Weak Alignment";
  };

  return (
    <div className="interpret-card" id="physics-validation">
      <div className="interpret-card-header">
        <div className="interpret-card-title">
          <ShieldCheck size={16} />
          <span>Physics Validation</span>
        </div>
        <InfoTooltip title="Physics Validation">
          <p>Checks if the ML model&apos;s learned feature importances align with known piezoelectric physics.</p>
          <p><strong>Confirmed:</strong> The model learned relationships consistent with solid-state physics theory.</p>
          <p><strong>Violations:</strong> The model learned inverse or unexpected relationships — may indicate overfitting or data bias.</p>
          <p>A high alignment score increases confidence that the model has learned physically meaningful patterns.</p>
        </InfoTooltip>
      </div>

      <div className="interpret-card-body">
        {physicsLoading && (
          <div className="interpret-loading">
            <Loader2 size={20} className="spin" />
            <span>Validating physics...</span>
          </div>
        )}
        {physicsError && <div className="interpret-error">{physicsError}</div>}
        {physics && !physicsLoading && (
          <div className="physics-container">
            {/* Score Circle */}
            <div className="physics-score-section">
              <div
                className="physics-score-circle"
                style={{ borderColor: scoreColor(physics.alignment_score) }}
              >
                <span
                  className="physics-score-value"
                  style={{ color: scoreColor(physics.alignment_score) }}
                >
                  {physics.alignment_score.toFixed(0)}%
                </span>
                <span className="physics-score-label">
                  {scoreLabel(physics.alignment_score)}
                </span>
              </div>
              <div className="physics-score-meta">
                <span className="confirmed-count">
                  <CheckCircle2 size={14} />
                  {physics.confirmed}/{physics.total_checks} confirmed
                </span>
                {physics.violations.length > 0 && (
                  <span className="violation-count">
                    <AlertTriangle size={14} />
                    {physics.violations.length} violation{physics.violations.length > 1 ? "s" : ""}
                  </span>
                )}
                {physics.skipped.length > 0 && (
                  <span className="skipped-count">
                    {physics.skipped.length} skipped (features not in model)
                  </span>
                )}
              </div>
            </div>

            {/* Toggle details */}
            <button
              className="physics-toggle-btn"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {showDetails ? "Hide details" : "Show details"}
            </button>

            {showDetails && (
              <div className="physics-details">
                {/* Confirmed checks */}
                {physics.confirmed_checks.length > 0 && (
                  <div className="physics-section">
                    <h4 className="physics-section-title confirmed">
                      <CheckCircle2 size={14} /> Confirmed Physics
                    </h4>
                    {physics.confirmed_checks.map((c) => (
                      <div key={c.feature} className="physics-check-row confirmed">
                        <div className="physics-check-feature font-mono">{c.feature}</div>
                        <div className="physics-check-reason">{c.physics_reason}</div>
                        <div className="physics-check-meta">
                          Rank #{c.shap_rank} • Expected: {c.expected_effect}
                          {c.actual_effect !== "unknown" && ` • Actual: ${c.actual_effect}`}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Violations */}
                {physics.violations.length > 0 && (
                  <div className="physics-section">
                    <h4 className="physics-section-title violation">
                      <AlertTriangle size={14} /> Violations Detected
                    </h4>
                    {physics.violations.map((c) => (
                      <div key={c.feature} className="physics-check-row violation">
                        <div className="physics-check-feature font-mono">{c.feature}</div>
                        <div className="physics-check-reason">{c.physics_reason}</div>
                        <div className="physics-check-meta">
                          Rank #{c.shap_rank} • Expected: {c.expected_effect}
                          {c.actual_effect !== "unknown" && ` • Actual: ${c.actual_effect}`}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {!physics && !physicsLoading && !physicsError && (
          <div className="interpret-empty">
            Select a model to validate physics alignment
          </div>
        )}
      </div>
    </div>
  );
}
