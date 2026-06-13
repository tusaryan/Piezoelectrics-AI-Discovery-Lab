"use client";

/**
 * BatchUpload — CSV batch prediction with drag-drop + from existing dataset.
 * Multi-target support: predicts d33, tc, and hardness with per-target models.
 * Shows tabular results preview + download.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileSpreadsheet,
  Upload,
  Download,
  AlertCircle,
  Database,
  ChevronDown,
  ChevronUp,
  Table,
} from "lucide-react";
import {
  predictBatchCSV,
  predictBatchFromDataset,
  getBatchDownloadUrl,
  type BatchResultRow,
} from "@/lib/api/predictions";
import { listDatasets, type DatasetSummary } from "@/lib/api/datasets";
import { usePredictStore } from "@/lib/store/predictStore";

export default function BatchUpload() {
  const {
    targetModels,
    batchResult,
    setBatchResult,
    batchLoading,
    setBatchLoading,
    batchError,
    setBatchError,
  } = usePredictStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [batchMode, setBatchMode] = useState<"csv" | "dataset">("csv");
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [showTable, setShowTable] = useState(true);

  // Build model IDs dict from per-target selection
  const hasAnyModel = targetModels.d33 || targetModels.tc || targetModels.vickers_hardness;

  // Count how many targets are selected
  const selectedTargetCount = [targetModels.d33, targetModels.tc, targetModels.vickers_hardness].filter(Boolean).length;

  // Load datasets when switching to dataset mode
  useEffect(() => {
    if (batchMode !== "dataset") return;
    setLoadingDatasets(true);
    listDatasets()
      .then((data) => setDatasets(data.filter((d) => d.status === "ready")))
      .catch(() => setDatasets([]))
      .finally(() => setLoadingDatasets(false));
  }, [batchMode]);

  const handleFile = useCallback(
    async (file: File) => {
      if (!hasAnyModel) {
        setBatchError("Please select at least one model");
        return;
      }
      if (!file.name.endsWith(".csv")) {
        setBatchError("Only CSV files are supported");
        return;
      }
      setBatchLoading(true);
      setBatchError(null);
      setBatchResult(null);
      try {
        const result = await predictBatchCSV({ ...targetModels }, file);
        setBatchResult(result);
      } catch (err) {
        setBatchError(err instanceof Error ? err.message : "Batch prediction failed");
      } finally {
        setBatchLoading(false);
      }
    },
    [targetModels, hasAnyModel, setBatchLoading, setBatchError, setBatchResult],
  );

  const handleFromDataset = useCallback(async () => {
    if (!hasAnyModel) {
      setBatchError("Please select at least one model");
      return;
    }
    if (!selectedDatasetId) {
      setBatchError("Please select a dataset");
      return;
    }
    setBatchLoading(true);
    setBatchError(null);
    setBatchResult(null);
    try {
      const result = await predictBatchFromDataset({ ...targetModels }, selectedDatasetId);
      setBatchResult(result);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "Batch prediction failed");
    } finally {
      setBatchLoading(false);
    }
  }, [targetModels, hasAnyModel, selectedDatasetId, setBatchLoading, setBatchError, setBatchResult]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  // Format value with CI
  const formatVal = (val: number | null | undefined, lo: number | null | undefined, hi: number | null | undefined) => {
    if (val == null) return "—";
    const v = val.toFixed(1);
    if (lo != null && hi != null) {
      return `${v} [${lo.toFixed(1)}–${hi.toFixed(1)}]`;
    }
    return v;
  };

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <FileSpreadsheet size={16} /> Batch Prediction
        {hasAnyModel && (
          <span style={{ fontSize: 10, color: "var(--text-muted)", fontWeight: 400, marginLeft: 8 }}>
            ({selectedTargetCount} target{selectedTargetCount > 1 ? "s" : ""} selected)
          </span>
        )}
      </div>

      {!batchResult && (
        <>
          {/* Mode toggle */}
          <div className="batch-mode-toggle">
            <button
              className={`batch-mode-btn ${batchMode === "csv" ? "active" : ""}`}
              onClick={() => setBatchMode("csv")}
            >
              <Upload size={13} /> Upload CSV
            </button>
            <button
              className={`batch-mode-btn ${batchMode === "dataset" ? "active" : ""}`}
              onClick={() => setBatchMode("dataset")}
            >
              <Database size={13} /> From Dataset
            </button>
          </div>

          {batchMode === "csv" && (
            <>
              <div
                className={`batch-dropzone ${dragActive ? "active" : ""}`}
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="batch-dropzone-icon"><Upload size={22} /></div>
                <div className="batch-dropzone-text">
                  {batchLoading ? "Processing..." : "Drop CSV here or click to browse"}
                </div>
                <div className="batch-dropzone-hint">
                  Must contain a &quot;formula&quot; column. For composites, include
                  matrix_type, filler_wt_pct, etc.
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                  e.target.value = "";
                }}
              />
            </>
          )}

          {batchMode === "dataset" && (
            <div className="batch-dataset-picker">
              {loadingDatasets ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Loading datasets...</p>
              ) : datasets.length === 0 ? (
                <p style={{ fontSize: 13, color: "var(--text-muted)" }}>No ready datasets found. Upload one first.</p>
              ) : (
                <>
                  <div className="batch-dataset-select-wrapper">
                    <select
                      className="batch-dataset-select"
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                    >
                      <option value="">— Select a dataset —</option>
                      {datasets.map((ds) => (
                        <option key={ds.id} value={ds.id}>
                          {ds.display_name} ({ds.total_rows} rows)
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={14} className="batch-dataset-chevron" />
                  </div>
                  <button
                    className="predict-submit-btn"
                    style={{ marginTop: 12 }}
                    disabled={!selectedDatasetId || !hasAnyModel || batchLoading}
                    onClick={handleFromDataset}
                  >
                    {batchLoading ? "Processing..." : "Run Batch Prediction"}
                  </button>
                </>
              )}
            </div>
          )}
        </>
      )}

      {batchError && (
        <div className="formula-error" style={{ marginTop: 12 }}>
          <AlertCircle size={14} /> {batchError}
        </div>
      )}

      {batchResult && (
        <>
          {/* Summary stats */}
          <div className="batch-summary">
            <div className="batch-stat">
              <div className="batch-stat-value">{batchResult.total_rows}</div>
              <div className="batch-stat-label">Total Rows</div>
            </div>
            <div className="batch-stat">
              <div className="batch-stat-value success">{batchResult.success_count}</div>
              <div className="batch-stat-label">Predicted</div>
            </div>
            <div className="batch-stat">
              <div className="batch-stat-value error">{batchResult.error_count}</div>
              <div className="batch-stat-label">Errors</div>
            </div>
          </div>

          {/* Tabular results toggle */}
          {batchResult.results && batchResult.results.length > 0 && (
            <>
              <button
                onClick={() => setShowTable(!showTable)}
                style={{
                  display: "flex", alignItems: "center", gap: 6,
                  background: "none", border: "1px solid var(--border-color)",
                  color: "var(--text-primary)", cursor: "pointer",
                  fontSize: 12, padding: "6px 12px", borderRadius: 6,
                  width: "100%", justifyContent: "center",
                  marginTop: 8,
                }}
              >
                <Table size={13} />
                {showTable ? "Hide" : "Show"} Results Table
                {showTable ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>

              {showTable && (
                <div className="batch-results-table-wrapper" style={{
                  marginTop: 8, overflowX: "auto", borderRadius: 8,
                  border: "1px solid var(--border-color)",
                  maxHeight: 400, overflowY: "auto",
                }}>
                  <table className="batch-results-table" style={{
                    width: "100%", borderCollapse: "collapse", fontSize: 11,
                    fontFamily: "var(--font-mono, monospace)",
                  }}>
                    <thead>
                      <tr style={{
                        background: "var(--surface-hover)",
                        position: "sticky", top: 0, zIndex: 1,
                      }}>
                        <th style={thStyle}>#</th>
                        <th style={thStyle}>Formula</th>
                        {targetModels.d33 && <th style={thStyle}>d₃₃ (pC/N)</th>}
                        {targetModels.tc && <th style={thStyle}>Tc (°C)</th>}
                        {targetModels.vickers_hardness && <th style={thStyle}>HV</th>}
                        <th style={thStyle}>Top Use Case</th>
                        <th style={thStyle}>Score</th>
                        <th style={thStyle}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResult.results.map((row: BatchResultRow, i: number) => (
                        <tr key={i} style={{
                          borderBottom: "1px solid var(--border-color)",
                          background: row.prediction_status !== "success"
                            ? "rgba(239, 68, 68, 0.05)" : "transparent",
                        }}>
                          <td style={tdStyle}>{row.uid ?? i + 1}</td>
                          <td style={{ ...tdStyle, fontWeight: 500, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {row.formula}
                          </td>
                          {targetModels.d33 && (
                            <td style={tdStyle}>
                              {formatVal(row.d33_predicted, row.d33_ci_lower, row.d33_ci_upper)}
                            </td>
                          )}
                          {targetModels.tc && (
                            <td style={tdStyle}>
                              {formatVal(row.tc_predicted, row.tc_ci_lower, row.tc_ci_upper)}
                            </td>
                          )}
                          {targetModels.vickers_hardness && (
                            <td style={tdStyle}>
                              {formatVal(row.hardness_predicted, row.hardness_ci_lower, row.hardness_ci_upper)}
                            </td>
                          )}
                          <td style={{ ...tdStyle, maxWidth: 150, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {row.top_use_case || "—"}
                          </td>
                          <td style={tdStyle}>
                            {row.use_case_score != null ? (
                              <span style={{
                                background: row.use_case_score >= 70 ? "#10B98120" : row.use_case_score >= 45 ? "#F59E0B20" : "#EF444420",
                                color: row.use_case_score >= 70 ? "#10B981" : row.use_case_score >= 45 ? "#F59E0B" : "#EF4444",
                                padding: "1px 5px", borderRadius: 3, fontSize: 10, fontWeight: 600,
                              }}>
                                {row.use_case_score}%
                              </span>
                            ) : "—"}
                          </td>
                          <td style={tdStyle}>
                            <span style={{
                              fontSize: 9, fontWeight: 600, textTransform: "uppercase",
                              color: row.prediction_status === "success" ? "#10B981" : "#EF4444",
                            }}>
                              {row.prediction_status === "success" ? "✓" : "✗"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {/* Action buttons */}
          <a href={getBatchDownloadUrl(batchResult.batch_id)} className="batch-download-btn" download style={{ marginTop: 10 }}>
            <Download size={16} /> Download Results CSV
          </a>

          <button
            className="comparison-action-btn"
            style={{ marginTop: 8, width: "100%", justifyContent: "center" }}
            onClick={() => { setBatchResult(null); setBatchError(null); }}
          >
            Run Another Batch
          </button>
        </>
      )}
    </div>
  );
}

/* Table cell styles */
const thStyle: React.CSSProperties = {
  padding: "6px 8px",
  textAlign: "left",
  fontWeight: 600,
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.3px",
  color: "var(--text-secondary)",
  borderBottom: "2px solid var(--border-color)",
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  padding: "5px 8px",
  color: "var(--text-primary)",
  whiteSpace: "nowrap",
};
