"use client";

/**
 * Piezo.AI — Dataset List
 * =========================
 * Multi-dataset manager: view, rename, copy, delete (single + bulk).
 * Actions (rename, copy, delete) are behind a three-dot kebab menu.
 */

import { useCallback, useEffect, useState, useRef } from "react";
import {
  Database,
  Upload,
  Trash2,
  Play,
  Eye,
  Loader2,
  FileText,
  Calendar,
  Copy,
  Pencil,
  Check,
  X,
  CheckSquare,
  Square,
  MoreVertical,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import {
  listDatasets,
  deleteDataset,
  renameDataset,
  copyDataset,
  bulkDeleteDatasets,
} from "@/lib/api/datasets";

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

/* ---------- Kebab Menu ---------- */

interface KebabMenuProps {
  dsId: string;
  dsName: string;
  onRename: (id: string, name: string) => void;
  onCopy: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  copyingId: string | null;
  deletingId: string | null;
}

function KebabMenu({ dsId, dsName, onRename, onCopy, onDelete, copyingId, deletingId }: KebabMenuProps) {
  const [open, setOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [copied, setCopied] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
        setConfirmDelete(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const handleCopy = async () => {
    onCopy(dsId, dsName);
    setCopied(true);
    setTimeout(() => { setCopied(false); setOpen(false); }, 800);
  };

  const handleDelete = () => {
    onDelete(dsId);
    setTimeout(() => { setOpen(false); setConfirmDelete(false); }, 300);
  };

  return (
    <div className="kebab-wrapper" ref={menuRef}>
      <button
        className="kebab-trigger"
        onClick={(e) => { e.stopPropagation(); setOpen(!open); setConfirmDelete(false); }}
        aria-label="More actions"
      >
        <MoreVertical size={16} />
      </button>

      {open && (
        <div className="kebab-popover">
          {/* Rename */}
          <button
            className="kebab-item"
            onClick={() => { onRename(dsId, dsName); setOpen(false); }}
          >
            <Pencil size={14} /> Rename
          </button>

          {/* Copy */}
          <button
            className="kebab-item"
            onClick={handleCopy}
            disabled={copyingId === dsId}
          >
            {copied ? (
              <><Check size={14} className="kebab-check" /> Copied!</>
            ) : copyingId === dsId ? (
              <><Loader2 size={14} className="spin" /> Copying...</>
            ) : (
              <><Copy size={14} /> Copy</>
            )}
          </button>

          {/* Delete */}
          {confirmDelete ? (
            <div className="kebab-confirm">
              <span className="kebab-confirm-text">Delete?</span>
              <button
                className="kebab-item kebab-danger"
                onClick={handleDelete}
                disabled={deletingId === dsId}
              >
                {deletingId === dsId ? <Loader2 size={12} className="spin" /> : "Yes"}
              </button>
              <button
                className="kebab-item"
                onClick={() => setConfirmDelete(false)}
              >
                No
              </button>
            </div>
          ) : (
            <button
              className="kebab-item kebab-danger"
              onClick={() => setConfirmDelete(true)}
            >
              <Trash2 size={14} /> Delete
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------- Component ---------- */

export default function DatasetList() {
  const {
    datasets,
    isLoadingDatasets,
    setDatasets,
    setIsLoadingDatasets,
    setActiveDatasetId,
    startWizard,
    removeDatasetFromList,
    setWizardStep,
  } = useDatasetStore();

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [copyingId, setCopyingId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [bulkMode, setBulkMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);

  const refreshDatasets = useCallback(async () => {
    setIsLoadingDatasets(true);
    try {
      const data = await listDatasets();
      setDatasets(data);
    } catch {
      setDatasets([]);
    } finally {
      setIsLoadingDatasets(false);
    }
  }, [setDatasets, setIsLoadingDatasets]);

  useEffect(() => { refreshDatasets(); }, [refreshDatasets]);

  const handleOpen = useCallback(
    (id: string, status: string) => {
      setActiveDatasetId(id);
      if (status === "pending") setWizardStep("review");
    },
    [setActiveDatasetId, setWizardStep],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      setDeletingId(id);
      try {
        await deleteDataset(id);
        removeDatasetFromList(id);
      } catch { /* ignore */ }
      finally { setDeletingId(null); }
    },
    [removeDatasetFromList],
  );

  const handleCopy = useCallback(
    async (id: string, name: string) => {
      setCopyingId(id);
      try {
        await copyDataset(id, `${name} (Copy)`);
        await refreshDatasets();
      } catch { /* ignore */ }
      finally { setCopyingId(null); }
    },
    [refreshDatasets],
  );

  const handleRenameStart = useCallback((id: string, currentName: string) => {
    setRenamingId(id);
    setRenameValue(currentName);
  }, []);

  const handleRenameSubmit = useCallback(
    async (id: string) => {
      if (!renameValue.trim()) return;
      try {
        await renameDataset(id, renameValue.trim());
        await refreshDatasets();
      } catch { /* ignore */ }
      finally { setRenamingId(null); }
    },
    [renameValue, refreshDatasets],
  );

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    if (selectedIds.size === datasets.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(datasets.map((d) => d.id)));
    }
  }, [selectedIds.size, datasets]);

  const handleBulkDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    setBulkDeleting(true);
    try {
      await bulkDeleteDatasets(Array.from(selectedIds));
      await refreshDatasets();
      setSelectedIds(new Set());
      setBulkMode(false);
      setConfirmBulkDelete(false);
    } catch { /* ignore */ }
    finally { setBulkDeleting(false); }
  }, [selectedIds, refreshDatasets]);

  const handleNewUpload = useCallback(() => { startWizard(); }, [startWizard]);

  const isAllSelected = selectedIds.size === datasets.length && datasets.length > 0;

  return (
    <div className="dataset-list">
      <div className="dataset-list-header">
        <div className="dataset-list-title">
          <Database size={20} />
          <h2>Your Datasets</h2>
          {datasets.length > 0 && (
            <span className="dataset-count-badge">{datasets.length}</span>
          )}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {datasets.length > 1 && (
            <button
              className={`btn-ghost btn-sm ${bulkMode ? "btn-active" : ""}`}
              onClick={() => {
                setBulkMode(!bulkMode);
                setSelectedIds(new Set());
                setConfirmBulkDelete(false);
              }}
            >
              {bulkMode ? "Cancel" : "Manage"}
            </button>
          )}
          <button className="btn-primary" onClick={handleNewUpload}>
            <Upload size={16} />
            Upload New Dataset
          </button>
        </div>
      </div>

      {/* Bulk actions bar */}
      {bulkMode && datasets.length > 0 && (
        <div className="dataset-bulk-bar">
          <button className="btn-ghost btn-sm" onClick={toggleSelectAll}>
            {isAllSelected ? (
              <><CheckSquare size={14} /> Deselect All</>
            ) : (
              <><Square size={14} /> Select All</>
            )}
          </button>
          <span className="dataset-bulk-count">
            {selectedIds.size} of {datasets.length} selected
          </span>
          {selectedIds.size > 0 && !confirmBulkDelete && (
            <button
              className="btn-ghost btn-sm btn-danger-text"
              onClick={() => setConfirmBulkDelete(true)}
            >
              <Trash2 size={14} />
              {isAllSelected ? "Delete All" : "Delete Selected"}
            </button>
          )}
          {confirmBulkDelete && (
            <div className="dataset-bulk-confirm">
              <span>
                {isAllSelected
                  ? `Delete all ${datasets.length} dataset(s)?`
                  : `Delete ${selectedIds.size} dataset(s)?`}
              </span>
              <button
                className="btn-ghost btn-sm btn-danger-text"
                onClick={handleBulkDelete}
                disabled={bulkDeleting}
              >
                {bulkDeleting ? <Loader2 size={12} className="spin" /> : "Yes, Delete"}
              </button>
              <button
                className="btn-ghost btn-sm"
                onClick={() => setConfirmBulkDelete(false)}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      {isLoadingDatasets && (
        <div className="dataset-list-loading">
          <Loader2 size={24} className="spin" />
          <p>Loading datasets...</p>
        </div>
      )}

      {!isLoadingDatasets && datasets.length === 0 && (
        <div className="dataset-list-empty">
          <div className="dataset-list-empty-icon"><FileText size={48} /></div>
          <h3>No datasets uploaded yet</h3>
          <p>Upload a CSV file to get started with your piezoelectric material data.</p>
          <button className="btn-primary" onClick={handleNewUpload}>
            <Upload size={16} /> Upload Your First Dataset
          </button>
        </div>
      )}

      {!isLoadingDatasets && datasets.length > 0 && (
        <div className="dataset-cards-grid">
          {datasets.map((ds) => (
            <div
              key={ds.id}
              className={`dataset-card ${bulkMode && selectedIds.has(ds.id) ? "selected" : ""} ${bulkMode ? "bulk-mode" : ""}`}
            >
              {bulkMode && (
                <button
                  className="dataset-card-checkbox"
                  onClick={() => toggleSelect(ds.id)}
                >
                  {selectedIds.has(ds.id) ? (
                    <CheckSquare size={16} style={{ color: "var(--primary)" }} />
                  ) : (
                    <Square size={16} />
                  )}
                </button>
              )}
              <div className="dataset-card-header">
                {renamingId === ds.id ? (
                  <div className="dataset-rename-row">
                    <input
                      className="dataset-rename-input"
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleRenameSubmit(ds.id)}
                      autoFocus
                    />
                    <button className="btn-ghost btn-sm" onClick={() => handleRenameSubmit(ds.id)}>
                      <Check size={12} />
                    </button>
                    <button className="btn-ghost btn-sm" onClick={() => setRenamingId(null)}>
                      <X size={12} />
                    </button>
                  </div>
                ) : (
                  <div className="dataset-card-name">{ds.display_name}</div>
                )}
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
                  <Calendar size={12} /> {timeAgo(ds.uploaded_at)}
                </span>

                <div className="dataset-card-actions">
                  {/* View — always visible */}
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

                  {/* Three-dot kebab menu — hidden in bulk mode */}
                  {!bulkMode && (
                    <KebabMenu
                      dsId={ds.id}
                      dsName={ds.display_name}
                      onRename={handleRenameStart}
                      onCopy={handleCopy}
                      onDelete={handleDelete}
                      copyingId={copyingId}
                      deletingId={deletingId}
                    />
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

