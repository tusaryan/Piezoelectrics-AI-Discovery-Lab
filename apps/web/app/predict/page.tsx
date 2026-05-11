"use client";

/**
 * Predict Page — Unified prediction: bulk ceramics + composites + hardness.
 * Supports per-target model selection (predict d33, tc, hardness independently).
 */

import { useCallback, useEffect, useState } from "react";
import {
  Zap,
  Atom,
  FileSpreadsheet,
  GitCompareArrows,
  Loader2,
  CheckCircle,
  AlertCircle,
  ShieldAlert,
} from "lucide-react";
import { predictSingle } from "@/lib/api/predictions";
import type { PredictResponse } from "@/lib/api/predictions";
import { usePredictStore } from "@/lib/store/predictStore";
import FormulaInput from "@/components/predict/FormulaInput";
import ModelSelector from "@/components/predict/ModelSelector";
import CompositeFields from "@/components/predict/CompositeFields";
import PredictionGauges from "@/components/predict/PredictionGauges";
import UseCaseCard from "@/components/predict/UseCaseCard";
import ComparisonTable from "@/components/predict/ComparisonTable";
import BatchUpload from "@/components/predict/BatchUpload";

export default function PredictPage() {
  const {
    activeTab,
    setActiveTab,
    formula,
    formulaValidation,
    formulaValidating,
    targetModels,
    isComposite,
    compositeParams,
    prediction,
    predicting,
    setPrediction,
    setPredicting,
    predictionError,
    setPredictionError,
    addToComparison,
    comparisonList,
    comparisonJustAdded,
    setComparisonJustAdded,
  } = usePredictStore();

  // "Added!" confirmation timer
  useEffect(() => {
    if (!comparisonJustAdded) return;
    const t = setTimeout(() => setComparisonJustAdded(false), 1800);
    return () => clearTimeout(t);
  }, [comparisonJustAdded, setComparisonJustAdded]);

  const hasAnyModel = Object.values(targetModels).some(Boolean);
  const formulaReady = formula.trim() && formulaValidation?.is_valid && !formulaValidating;
  const canPredict = formulaReady && hasAnyModel;

  // Determine why predict is blocked (for error message)
  const getBlockReason = (): string | null => {
    if (!formula.trim()) return null; // empty = no message
    if (formulaValidating) return "Validating formula...";
    if (formulaValidation && !formulaValidation.is_valid)
      return formulaValidation.error || "Invalid formula — fix before predicting";
    if (!hasAnyModel) return "Select at least one model to predict";
    return null;
  };
  const blockReason = getBlockReason();

  const handlePredict = useCallback(async () => {
    if (!canPredict) return;

    setPredicting(true);
    setPredictionError(null);
    setPrediction(null);

    // Predict with each selected model and merge results
    try {
      const merged: Partial<PredictResponse> = {
        formula,
        is_composite: isComposite,
        status: "success",
        notes: null,
        d33: null,
        tc: null,
        hardness: null,
        use_case: null,
        composite_params: null,
      };

      const targets = ["d33", "tc", "vickers_hardness"] as const;
      for (const target of targets) {
        const modelId = targetModels[target];
        if (!modelId) continue;

        const result = await predictSingle(
          formula,
          modelId,
          isComposite ? compositeParams : undefined,
        );

        if (result.status !== "success") {
          merged.status = result.status;
          merged.notes = result.notes;
          break;
        }

        if (result.d33?.value != null) merged.d33 = result.d33;
        if (result.tc?.value != null) merged.tc = result.tc;
        if (result.hardness?.value != null) merged.hardness = result.hardness;
        if (result.use_case && !merged.use_case) merged.use_case = result.use_case;
        if (result.composite_params) merged.composite_params = result.composite_params;
      }

      setPrediction(merged as PredictResponse);
    } catch (err) {
      setPredictionError(
        err instanceof Error ? err.message : "Prediction failed",
      );
    } finally {
      setPredicting(false);
    }
  }, [
    canPredict, formula, targetModels, isComposite, compositeParams,
    setPredicting, setPredictionError, setPrediction,
  ]);

  const handleAddToComparison = useCallback(() => {
    if (!prediction || prediction.status !== "success") return;
    addToComparison({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      formula: prediction.formula,
      prediction,
      timestamp: new Date().toISOString(),
    });
  }, [prediction, addToComparison]);

  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon"><Zap size={22} /></div>
        <div className="page-header-text">
          <h1>Predict</h1>
          <p>Unified prediction: bulk ceramics + composites + hardness</p>
        </div>
      </div>

      {/* Tab Bar */}
      <div className="predict-tabs">
        <button className={`predict-tab ${activeTab === "single" ? "active" : ""}`} onClick={() => setActiveTab("single")}>
          <Atom size={15} /> Single Prediction
        </button>
        <button className={`predict-tab ${activeTab === "batch" ? "active" : ""}`} onClick={() => setActiveTab("batch")}>
          <FileSpreadsheet size={15} /> Batch Processing
        </button>
        <button className={`predict-tab ${activeTab === "comparison" ? "active" : ""}`} onClick={() => setActiveTab("comparison")}>
          <GitCompareArrows size={15} /> Comparison
          {comparisonList.length > 0 && (
            <span style={{ fontSize: 10, fontWeight: 700, padding: "1px 5px", borderRadius: 999, background: "var(--primary-glow)", color: "var(--primary)", marginLeft: 2 }}>
              {comparisonList.length}
            </span>
          )}
        </button>
      </div>

      <div className="predict-layout">
        {activeTab === "single" && (
          <div className="predict-form">
            <div className="predict-input-panel">
              <ModelSelector />
              <div className="predict-card">
                <div className="predict-card-title"><Atom size={16} /> Formula</div>
                <FormulaInput />
              </div>
              <CompositeFields />

              {/* Block reason message */}
              {blockReason && (
                <div className="predict-block-reason">
                  <ShieldAlert size={14} />
                  <span>{blockReason}</span>
                </div>
              )}

              <button
                id="predict-btn"
                className={`predict-submit-btn ${predicting ? "loading" : ""}`}
                disabled={!canPredict || predicting}
                onClick={handlePredict}
              >
                {predicting ? (
                  <><Loader2 size={16} className="spin" /> Predicting...</>
                ) : (
                  <><Zap size={16} /> Predict Properties</>
                )}
              </button>
            </div>

            <div className="predict-results-panel">
              {predictionError && (
                <div className="predict-card">
                  <div className="formula-error"><AlertCircle size={16} /><span>{predictionError}</span></div>
                </div>
              )}

              {prediction && prediction.status !== "success" && (
                <div className="predict-card">
                  <div className="formula-error">
                    <AlertCircle size={16} />
                    <span>{prediction.notes || "Prediction could not be completed"}</span>
                  </div>
                </div>
              )}

              <PredictionGauges />
              <UseCaseCard />

              {prediction && prediction.status === "success" && (
                <button
                  className={`comparison-action-btn ${comparisonJustAdded ? "added" : "primary"}`}
                  style={{ width: "100%", justifyContent: "center", padding: "10px 16px" }}
                  onClick={handleAddToComparison}
                  disabled={comparisonJustAdded}
                >
                  {comparisonJustAdded ? (
                    <><CheckCircle size={14} /> Added to Comparison!</>
                  ) : (
                    <><GitCompareArrows size={14} /> Add to Comparison</>
                  )}
                </button>
              )}

              {!prediction && !predictionError && (
                <div className="predict-card">
                  <div className="predict-empty-state">
                    <div className="predict-empty-icon"><GaugeIcon size={24} /></div>
                    <h3>Results will appear here</h3>
                    <p>Select models, enter a formula, and click Predict to see d₃₃, Tc, or hardness predictions with confidence intervals.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "batch" && (
          <div style={{ maxWidth: 700 }}>
            <ModelSelector />
            <div style={{ marginTop: 16 }}><BatchUpload /></div>
          </div>
        )}

        {activeTab === "comparison" && <ComparisonTable />}
      </div>
    </div>
  );
}

function GaugeIcon({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 14 4-4" /><path d="M3.34 19a10 10 0 1 1 17.32 0" />
    </svg>
  );
}
