"use client";

import { useEffect, useState } from "react";
import {
  FileText,
  Download,
  Loader2,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Trash2,
  Info,
  CheckSquare,
  Square,
} from "lucide-react";
import type { PredictionHistoryItem } from "@/lib/api/dashboard";
import { getReportDownloadUrl, getLlmStatus } from "@/lib/api/dashboard";
import { useDashboardStore } from "@/lib/store/dashboardStore";

interface ReportGeneratorProps {
  predictionHistory: PredictionHistoryItem[];
  models: { id: string; display_name: string; target: string }[];
}

/** Format composite params into a short string */
function formatCompositeInfo(p: PredictionHistoryItem): string {
  if (!p.is_composite || !p.composite_params) return "";
  const cp = p.composite_params;
  const parts: string[] = [];
  if (cp.matrix_type && cp.matrix_type !== "none") parts.push(String(cp.matrix_type));
  if (cp.filler_wt_pct && Number(cp.filler_wt_pct) > 0) parts.push(`${cp.filler_wt_pct}wt%`);
  if (cp.particle_morphology && cp.particle_morphology !== "none") parts.push(String(cp.particle_morphology));
  if (cp.particle_size_nm) parts.push(`${cp.particle_size_nm}nm`);
  return parts.join(" · ");
}

/** Format exact datetime (e.g. "11 May 2026, 17:32:45") */
function formatExactTime(isoStr: string): string {
  const d = new Date(isoStr);
  return d.toLocaleDateString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
  }) + ", " + d.toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

export default function ReportGenerator({ predictionHistory, models }: ReportGeneratorProps) {
  const { generateReport, reportGenerating, lastReport, error, deletePrediction, bulkDeletePredictions } = useDashboardStore();

  // Report options
  const [includeR2Rmse, setIncludeR2Rmse] = useState(true);
  const [includePredActual, setIncludePredActual] = useState(true);
  const [includeShap, setIncludeShap] = useState(false);
  const [includeAiInsight, setIncludeAiInsight] = useState(false);
  const [includeMaterialInsight, setIncludeMaterialInsight] = useState(false);

  // Prediction selection
  const [selectedPredIds, setSelectedPredIds] = useState<Set<string>>(new Set());
  const [showPredHistory, setShowPredHistory] = useState(false);

  // Bulk delete confirmation
  const [showBulkDeleteConfirm, setShowBulkDeleteConfirm] = useState(false);

  // LLM config status — fetched dynamically from backend
  const [llmStatus, setLlmStatus] = useState<{
    configured: boolean;
    provider: string;
    model: string;
  } | null>(null);

  // Fetch LLM config on mount
  useEffect(() => {
    getLlmStatus()
      .then(setLlmStatus)
      .catch(() => setLlmStatus({ configured: false, provider: "none", model: "" }));
  }, []);

  const llmConfigured = llmStatus?.configured ?? false;

  function togglePrediction(id: string) {
    setSelectedPredIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAllPredictions() {
    setSelectedPredIds(new Set(predictionHistory.map((p) => p.id)));
  }

  function deselectAllPredictions() {
    setSelectedPredIds(new Set());
  }

  const allSelected = predictionHistory.length > 0 && selectedPredIds.size === predictionHistory.length;

  async function handleGenerate() {
    // Collect all member_ids from selected grouped predictions
    const allMemberIds: string[] = [];
    for (const pid of Array.from(selectedPredIds)) {
      const pred = predictionHistory.find(p => p.id === pid);
      if (pred?.member_ids?.length) {
        allMemberIds.push(...pred.member_ids);
      } else {
        allMemberIds.push(pid);
      }
    }
    await generateReport({
      include_r2_rmse: includeR2Rmse,
      include_predicted_vs_actual: includePredActual,
      include_shap_summary: includeShap,
      include_ai_insight: includeAiInsight,
      include_material_insight: includeMaterialInsight,
      selected_prediction_ids: allMemberIds,
      selected_model_ids: models.map((m) => m.id),
    });
  }

  // Individual delete — delete ALL member rows in the group
  async function handleDeletePrediction(id: string) {
    const pred = predictionHistory.find(p => p.id === id);
    const memberIds = pred?.member_ids?.length ? pred.member_ids : [id];
    if (memberIds.length > 1) {
      await bulkDeletePredictions(memberIds);
    } else {
      await deletePrediction(id);
    }
    setSelectedPredIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  // Bulk delete — requires confirmation, deletes all member rows
  async function handleBulkDeleteConfirmed() {
    const allMemberIds: string[] = [];
    for (const pid of Array.from(selectedPredIds)) {
      const pred = predictionHistory.find(p => p.id === pid);
      if (pred?.member_ids?.length) {
        allMemberIds.push(...pred.member_ids);
      } else {
        allMemberIds.push(pid);
      }
    }
    if (allMemberIds.length === 0) return;
    await bulkDeletePredictions(allMemberIds);
    setSelectedPredIds(new Set());
    setShowBulkDeleteConfirm(false);
  }

  return (
    <div className="dashboard-section report-generator">
      <div className="report-header">
        <h2 className="section-title">
          <FileText size={18} /> Report Generation
        </h2>
      </div>
      <p className="section-description">
        Generate a premium PDF report with model performance, charts, and insights.
      </p>

      {/* Report Options */}
      <div className="report-options">
        <label className="report-checkbox">
          <input
            type="checkbox"
            checked={includeR2Rmse}
            onChange={() => setIncludeR2Rmse(!includeR2Rmse)}
          />
          <span>R² / RMSE Comparison Charts</span>
        </label>

        <label className="report-checkbox">
          <input
            type="checkbox"
            checked={includePredActual}
            onChange={() => setIncludePredActual(!includePredActual)}
          />
          <span>Model Performance Overview</span>
        </label>

        <label className="report-checkbox">
          <input
            type="checkbox"
            checked={includeShap}
            onChange={() => setIncludeShap(!includeShap)}
          />
          <span>SHAP Analysis Summary</span>
        </label>

        <label className="report-checkbox">
          <input
            type="checkbox"
            checked={includeAiInsight}
            onChange={() => setIncludeAiInsight(!includeAiInsight)}
          />
          <span>AI Insight</span>
          {!llmConfigured && (
            <span className="report-badge-warn">
              <AlertTriangle size={12} /> Not configured
            </span>
          )}
          {llmConfigured && llmStatus && (
            <span className="report-badge-configured">
              <CheckCircle size={12} /> {llmStatus.provider}/{llmStatus.model}
            </span>
          )}
        </label>

        <label className="report-checkbox">
          <input
            type="checkbox"
            checked={includeMaterialInsight}
            onChange={() => setIncludeMaterialInsight(!includeMaterialInsight)}
          />
          <span>Material Prediction Insights</span>
        </label>
      </div>

      {/* AI Insight notice — only when not configured */}
      {includeAiInsight && !llmConfigured && (
        <div className="report-notice">
          <Info size={14} />
          <span>
            AI insight requires an LLM API key. Configure <code>LLM_PROVIDER</code>,{" "}
            <code>LLM_API_KEY</code>, and <code>LLM_MODEL</code> in your <code>.env</code> file.
            Supported providers: <code>openai</code>, <code>anthropic</code>, <code>google</code>, <code>ollama</code>.
          </span>
        </div>
      )}

      {/* Prediction History Selector */}
      {includeMaterialInsight && (
        <div className="prediction-selector">
          <button
            className="prediction-selector-toggle"
            onClick={() => setShowPredHistory(!showPredHistory)}
          >
            Select Predictions ({selectedPredIds.size} selected)
            {showPredHistory ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {showPredHistory && (
            <div className="prediction-list">
              {/* Top actions bar — always visible */}
              <div className="prediction-bulk-bar">
                <div className="prediction-bulk-left">
                  <button
                    className="prediction-select-all-btn"
                    onClick={allSelected ? deselectAllPredictions : selectAllPredictions}
                    title={allSelected ? "Deselect all" : "Select all"}
                  >
                    {allSelected ? <CheckSquare size={14} /> : <Square size={14} />}
                    {allSelected ? "Deselect All" : "Select All"}
                  </button>
                  <span className="prediction-bulk-count">
                    {selectedPredIds.size} of {predictionHistory.length}
                  </span>
                </div>
                {selectedPredIds.size > 0 && !showBulkDeleteConfirm && (
                  <button
                    className="prediction-bulk-delete"
                    onClick={() => setShowBulkDeleteConfirm(true)}
                    title="Delete selected predictions"
                  >
                    <Trash2 size={13} /> Delete Selected
                  </button>
                )}
                {showBulkDeleteConfirm && (
                  <div className="prediction-bulk-confirm">
                    <span>Delete {selectedPredIds.size} predictions?</span>
                    <button className="confirm-yes" onClick={handleBulkDeleteConfirmed}>
                      Yes, Delete
                    </button>
                    <button className="confirm-no" onClick={() => setShowBulkDeleteConfirm(false)}>
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              {predictionHistory.length === 0 ? (
                <div className="prediction-empty">No predictions in history</div>
              ) : (
                predictionHistory.slice(0, 50).map((p) => {
                  const compositeInfo = formatCompositeInfo(p);
                  return (
                    <div key={p.id} className="prediction-item">
                      <label className="prediction-item-label">
                        <input
                          type="checkbox"
                          checked={selectedPredIds.has(p.id)}
                          onChange={() => togglePrediction(p.id)}
                        />
                        <div className="prediction-item-content">
                          <div className="prediction-item-top">
                            <span className="prediction-formula">{p.formula}</span>
                            <span className={`prediction-type-badge ${p.is_composite ? "composite" : "bulk"}`}>
                              {p.is_composite ? "Composite" : "Bulk"}
                            </span>
                          </div>
                          {/* Composite properties row */}
                          {compositeInfo && (
                            <div className="prediction-composite-info">
                              {compositeInfo}
                            </div>
                          )}
                          <div className="prediction-item-bottom">
                            <span className="prediction-values">
                              {p.d33_predicted != null ? `d₃₃=${p.d33_predicted.toFixed(1)}` : ""}
                              {p.tc_predicted != null ? ` Tc=${p.tc_predicted.toFixed(1)}` : ""}
                              {p.hardness_predicted != null ? ` HV=${p.hardness_predicted.toFixed(1)}` : ""}
                            </span>
                            <span className="prediction-date">
                              {formatExactTime(p.created_at)}
                            </span>
                          </div>
                        </div>
                      </label>
                      {/* Individual delete — instant, no confirmation */}
                      <button
                        className="prediction-delete-btn"
                        onClick={() => handleDeletePrediction(p.id)}
                        title="Delete prediction"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      )}

      {/* Generate Button */}
      <div className="report-actions">
        <button
          className="report-generate-btn"
          onClick={handleGenerate}
          disabled={reportGenerating}
        >
          {reportGenerating ? (
            <>
              <Loader2 size={16} className="spin" /> Generating…
            </>
          ) : (
            <>
              <FileText size={16} /> Generate Report
            </>
          )}
        </button>

        {lastReport && (
          <a
            className="report-download-btn"
            href={getReportDownloadUrl(lastReport.report_id)}
            download
          >
            <Download size={16} /> Download PDF
          </a>
        )}
      </div>

      {/* Success message */}
      {lastReport && (
        <div className="report-success">
          <CheckCircle size={16} />
          Report generated: {lastReport.filename}
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="report-error">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}
    </div>
  );
}
