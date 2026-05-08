"use client";

/**
 * Piezo.AI — Dataset List
 * =========================
 * Multi-dataset selector shown when no dataset is active.
 * Cards with status badges, row counts, and actions.
 */

import { useCallback, useEffect, useState } from "react";
import {
  Database,
  Upload,
  Trash2,
  Play,
  Eye,
  Loader2,
  FileText,
  Calendar,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import { listDatasets, deleteDataset } from "@/lib/api/datasets";

/* ---------- Helpers ---------- */

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/* ---------- Component ---------- */

export default function DatasetList() {
  const {
    datasets,
    isLoadingDatasets,
    setDatasets,
    setIsLoadingDatasets,
    setActiveDatasetId,
    setActiveDataset,
    startWizard,
    removeDatasetFromList,
    setWizardStep,
  } = useDatasetStore();

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  /* Load datasets on mount */
  useEffect(() => {
    setIsLoadingDatasets(true);
    listDatasets()
      .then(setDatasets)
      .catch(() => setDatasets([]))
      .finally(() => setIsLoadingDatasets(false));
  }, [setDatasets, setIsLoadingDatasets]);

  /* Handle view/continue */
  const handleOpen = useCallback(
    (id: string, status: string) => {
      setActiveDatasetId(id);
      if (status === "pending") {
        setWizardStep("review"); // Resume at review step
      }
    },
    [setActiveDatasetId, setWizardStep],
  );

  /* Handle delete */
  const handleDelete = useCallback(
    async (id: string) => {
      setDeletingId(id);
      try {
        await deleteDataset(id);
        removeDatasetFromList(id);
        setConfirmDeleteId(null);
      } catch {
        /* ignore */
      } finally {
        setDeletingId(null);
      }
    },
    [removeDatasetFromList],
  );

  /* Handle new upload */
  const handleNewUpload = useCallback(() => {
    startWizard();
  }, [startWizard]);

  return (
    <div className="dataset-list">
      {/* Header */}
      <div className="dataset-list-header">
        <div className="dataset-list-title">
          <Database size={20} />
          <h2>Your Datasets</h2>
          {datasets.length > 0 && (
            <span className="dataset-count-badge">{datasets.length}</span>
          )}
        </div>
        <button className="btn-primary" onClick={handleNewUpload}>
          <Upload size={16} />
          Upload New Dataset
        </button>
      </div>

      {/* Loading */}
      {isLoadingDatasets && (
        <div className="dataset-list-loading">
          <Loader2 size={24} className="spin" />
          <p>Loading datasets...</p>
        </div>
      )}

      {/* Empty state */}
      {!isLoadingDatasets && datasets.length === 0 && (
        <div className="dataset-list-empty">
          <div className="dataset-list-empty-icon">
            <FileText size={48} />
          </div>
          <h3>No datasets uploaded yet</h3>
          <p>Upload a CSV file to get started with your piezoelectric material data.</p>
          <button className="btn-primary" onClick={handleNewUpload}>
            <Upload size={16} />
            Upload Your First Dataset
          </button>
        </div>
      )}

      {/* Dataset cards */}
      {!isLoadingDatasets && datasets.length > 0 && (
        <div className="dataset-cards-grid">
          {datasets.map((ds) => (
            <div key={ds.id} className="dataset-card">
              <div className="dataset-card-header">
                <div className="dataset-card-name">{ds.display_name}</div>
                <span className={`dataset-status-badge ${ds.status}`}>
                  {ds.status === "ready" ? "Ready" : "Pending"}
                </span>
              </div>

              <div className="dataset-card-meta">
                <span className="dataset-card-stat">
                  {ds.total_rows} rows × {ds.total_columns} cols
                </span>
                {ds.has_composite_fields && (
                  <span className="dataset-card-tag">Composite</span>
                )}
              </div>

              <div className="dataset-card-footer">
                <span className="dataset-card-date">
                  <Calendar size={12} />
                  {timeAgo(ds.uploaded_at)}
                </span>

                <div className="dataset-card-actions">
                  <button
                    className="btn-ghost btn-sm"
                    onClick={() => handleOpen(ds.id, ds.status)}
                    title={ds.status === "ready" ? "View" : "Continue"}
                  >
                    {ds.status === "ready" ? (
                      <><Eye size={14} /> View</>
                    ) : (
                      <><Play size={14} /> Continue</>
                    )}
                  </button>

                  {confirmDeleteId === ds.id ? (
                    <div className="dataset-card-confirm">
                      <span>Delete?</span>
                      <button
                        className="btn-ghost btn-sm btn-danger-text"
                        onClick={() => handleDelete(ds.id)}
                        disabled={deletingId === ds.id}
                      >
                        {deletingId === ds.id ? <Loader2 size={12} className="spin" /> : "Yes"}
                      </button>
                      <button
                        className="btn-ghost btn-sm"
                        onClick={() => setConfirmDeleteId(null)}
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button
                      className="btn-ghost btn-sm btn-danger-text"
                      onClick={() => setConfirmDeleteId(ds.id)}
                      title="Delete dataset"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
