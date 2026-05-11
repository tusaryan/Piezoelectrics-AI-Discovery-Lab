"use client";

import { useState, useRef, useEffect } from "react";
import {
  BrainCircuit,
  Pencil,
  Trash2,
  Download,
  Star,
  Check,
  X,
  CheckSquare,
  Square,
  FileSpreadsheet,
  FileArchive,
  Package,
} from "lucide-react";
import type { DashboardModel } from "@/lib/api/dashboard";
import { getParsedDatasetUrl } from "@/lib/api/dashboard";
import { useDashboardStore } from "@/lib/store/dashboardStore";
import { APP_CONFIG } from "@/lib/constants";

interface ModelLibraryProps {
  models: DashboardModel[];
}

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃",
  tc: "Tc",
  vickers_hardness: "Hardness",
};

const TARGET_COLORS: Record<string, string> = {
  d33: "#4F46E5",
  tc: "#10B981",
  vickers_hardness: "#F59E0B",
};

const ALGO_LABELS: Record<string, string> = {
  xgboost: "XGBoost",
  random_forest: "Random Forest",
  svr: "SVR",
  lightgbm: "LightGBM",
  gradient_boosting: "GBR",
  decision_tree: "Decision Tree",
  ann: "ANN",
  stacking: "Stacking",
};

function formatExactTime(isoStr: string): string {
  const d = new Date(isoStr);
  return d.toLocaleDateString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
  }) + ", " + d.toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

export default function ModelLibrary({ models }: ModelLibraryProps) {
  const {
    renameModel,
    setDefaultModel,
    deleteModel,
    bulkDeleteModels,
    selectedModelIds,
    toggleModelSelection,
    selectAllModels,
    clearModelSelection,
  } = useDashboardStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [bulkMode, setBulkMode] = useState(false);
  const [downloadMenuId, setDownloadMenuId] = useState<string | null>(null);
  const downloadMenuRef = useRef<HTMLDivElement>(null);

  // Close download menu on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (downloadMenuRef.current && !downloadMenuRef.current.contains(e.target as Node)) {
        setDownloadMenuId(null);
      }
    }
    if (downloadMenuId) {
      document.addEventListener("mousedown", handleClick);
      return () => document.removeEventListener("mousedown", handleClick);
    }
  }, [downloadMenuId]);

  function startRename(model: DashboardModel) {
    setEditingId(model.id);
    setEditName(model.display_name);
  }

  async function submitRename(id: string) {
    if (editName.trim()) {
      await renameModel(id, editName.trim());
    }
    setEditingId(null);
  }

  function cancelRename() {
    setEditingId(null);
    setEditName("");
  }

  async function handleDelete(id: string) {
    await deleteModel(id);
    setConfirmDeleteId(null);
  }

  function handleDownloadParsed(model: DashboardModel) {
    const url = getParsedDatasetUrl(model.id);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${model.display_name}_parsed.csv`;
    a.click();
    setDownloadMenuId(null);
  }

  function handleDownloadModel(model: DashboardModel) {
    // Download model .joblib file via direct path or API
    const url = `${APP_CONFIG.api.baseUrl}/api/v1/dashboard/models/${model.id}/model-file`;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${model.display_name}_model.joblib`;
    a.click();
    setDownloadMenuId(null);
  }

  function handleDownloadBoth(model: DashboardModel) {
    handleDownloadParsed(model);
    setTimeout(() => {
      handleDownloadModel(model);
    }, 500);
  }

  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}m ${s.toFixed(0)}s`;
  }

  function formatR2(value: number): string {
    return value.toFixed(4);
  }

  function r2Color(value: number): string {
    if (value >= 0.8) return "#10B981";
    if (value >= 0.5) return "#F59E0B";
    return "#EF4444";
  }

  return (
    <div className="dashboard-section model-library">
      <div className="section-header">
        <h2 className="section-title">
          <BrainCircuit size={18} /> Trained Models
        </h2>
        <div className="section-header-actions">
          <span className="section-count">{models.length}</span>
          {models.length > 0 && (
            <button
              className={`manage-btn ${bulkMode ? "active" : ""}`}
              onClick={() => {
                setBulkMode(!bulkMode);
                if (bulkMode) clearModelSelection();
              }}
            >
              {bulkMode ? "Done" : "Manage"}
            </button>
          )}
        </div>
      </div>

      {bulkMode && models.length > 0 && (
        <div className="bulk-action-bar">
          <button className="bulk-select-btn" onClick={selectAllModels}>
            Select All
          </button>
          <button className="bulk-select-btn" onClick={clearModelSelection}>
            Deselect All
          </button>
          {selectedModelIds.size > 0 && (
            <button className="bulk-delete-btn" onClick={bulkDeleteModels}>
              <Trash2 size={14} />
              Delete {selectedModelIds.size === models.length ? "All" : `Selected (${selectedModelIds.size})`}
            </button>
          )}
        </div>
      )}

      {models.length === 0 ? (
        <div className="empty-state">
          <BrainCircuit size={32} />
          <p>No trained models yet</p>
          <p className="empty-state-hint">Train a model in the Model Studio</p>
        </div>
      ) : (
        <div className="model-cards-grid">
          {models.map((model) => (
            <div
              key={model.id}
              className={`model-card ${model.is_default ? "model-default" : ""} ${selectedModelIds.has(model.id) ? "model-selected" : ""}`}
            >
              {/* Bulk checkbox */}
              {bulkMode && (
                <button
                  className="model-checkbox"
                  onClick={() => toggleModelSelection(model.id)}
                >
                  {selectedModelIds.has(model.id) ? (
                    <CheckSquare size={16} />
                  ) : (
                    <Square size={16} />
                  )}
                </button>
              )}

              {/* Header */}
              <div className="model-card-header">
                <div
                  className="model-target-badge"
                  style={{ backgroundColor: TARGET_COLORS[model.target] || "#64748B" }}
                >
                  {TARGET_LABELS[model.target] || model.target}
                </div>
                {model.is_default && (
                  <span className="model-default-badge">
                    <Star size={12} /> Default
                  </span>
                )}
              </div>

              {/* Name (editable) */}
              <div className="model-name-row">
                {editingId === model.id ? (
                  <div className="model-rename-form">
                    <input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") submitRename(model.id);
                        if (e.key === "Escape") cancelRename();
                      }}
                      autoFocus
                      className="model-rename-input"
                    />
                    <button
                      className="rename-confirm"
                      onClick={() => submitRename(model.id)}
                    >
                      <Check size={14} />
                    </button>
                    <button className="rename-cancel" onClick={cancelRename}>
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <h3 className="model-name" title={model.display_name}>
                    {model.display_name}
                  </h3>
                )}
              </div>

              {/* Metrics */}
              <div className="model-metrics">
                <div className="metric">
                  <span className="metric-label">R²</span>
                  <span
                    className="metric-value"
                    style={{ color: r2Color(model.r2_score) }}
                  >
                    {formatR2(model.r2_score)}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">RMSE</span>
                  <span className="metric-value">{model.rmse.toFixed(2)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Algorithm</span>
                  <span className="metric-value metric-algo">
                    {ALGO_LABELS[model.algorithm] || model.algorithm}
                  </span>
                </div>
              </div>

              {/* Meta info */}
              <div className="model-meta">
                <span>Train: {model.n_train_samples} · Test: {model.n_test_samples}</span>
                <span>{formatDuration(model.training_duration_s)}</span>
                <span>{formatExactTime(model.created_at)}</span>
              </div>

              {/* UUID (truncated, for reference) */}
              <div className="model-uuid" title={model.id}>
                UUID: {model.id.slice(0, 8)}…
              </div>

              {/* Actions */}
              {!bulkMode && (
                <div className="model-actions">
                  <button
                    className="model-action-btn"
                    onClick={() => startRename(model)}
                    title="Rename"
                  >
                    <Pencil size={14} />
                  </button>
                  <button
                    className="model-action-btn model-action-default"
                    onClick={() => setDefaultModel(model.id)}
                    title="Set as default"
                    disabled={model.is_default}
                  >
                    <Star size={14} />
                  </button>

                  {/* Download dropdown */}
                  <div className="download-dropdown-wrapper" ref={downloadMenuId === model.id ? downloadMenuRef : undefined}>
                    <button
                      className="model-action-btn"
                      onClick={() => setDownloadMenuId(downloadMenuId === model.id ? null : model.id)}
                      title="Download"
                    >
                      <Download size={14} />
                    </button>
                    {downloadMenuId === model.id && (
                      <div className="download-dropdown">
                        <button onClick={() => handleDownloadParsed(model)}>
                          <FileSpreadsheet size={14} />
                          Parsed Dataset (.csv)
                        </button>
                        <button onClick={() => handleDownloadModel(model)}>
                          <FileArchive size={14} />
                          Model Weights (.joblib)
                        </button>
                        <button onClick={() => handleDownloadBoth(model)}>
                          <Package size={14} />
                          Download Both
                        </button>
                      </div>
                    )}
                  </div>

                  {confirmDeleteId === model.id ? (
                    <div className="delete-confirm-inline">
                      <button
                        className="delete-yes"
                        onClick={() => handleDelete(model.id)}
                      >
                        Yes
                      </button>
                      <button
                        className="delete-no"
                        onClick={() => setConfirmDeleteId(null)}
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button
                      className="model-action-btn model-action-delete"
                      onClick={() => setConfirmDeleteId(model.id)}
                      title="Delete"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
