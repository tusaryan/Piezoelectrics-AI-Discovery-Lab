"use client";

/**
 * BatchUpload — CSV batch prediction with drag-drop + from existing dataset.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileSpreadsheet,
  Upload,
  Download,
  AlertCircle,
  Database,
  ChevronDown,
} from "lucide-react";
import {
  predictBatchCSV,
  predictBatchFromDataset,
  getBatchDownloadUrl,
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

  // Get the first available selected model for batch
  const batchModelId = targetModels.d33 || targetModels.tc || targetModels.vickers_hardness;

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
      if (!batchModelId) {
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
        const result = await predictBatchCSV(batchModelId, file);
        setBatchResult(result);
      } catch (err) {
        setBatchError(err instanceof Error ? err.message : "Batch prediction failed");
      } finally {
        setBatchLoading(false);
      }
    },
    [batchModelId, setBatchLoading, setBatchError, setBatchResult],
  );

  const handleFromDataset = useCallback(async () => {
    if (!batchModelId) {
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
      const result = await predictBatchFromDataset(batchModelId, selectedDatasetId);
      setBatchResult(result);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "Batch prediction failed");
    } finally {
      setBatchLoading(false);
    }
  }, [batchModelId, selectedDatasetId, setBatchLoading, setBatchError, setBatchResult]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <FileSpreadsheet size={16} /> Batch Prediction
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
                    disabled={!selectedDatasetId || !batchModelId || batchLoading}
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

          <a href={getBatchDownloadUrl(batchResult.batch_id)} className="batch-download-btn" download>
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
