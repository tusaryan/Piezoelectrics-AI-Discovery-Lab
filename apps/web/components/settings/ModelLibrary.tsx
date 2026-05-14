"use client";

import { useEffect, useState } from "react";
import { Brain, Star, Trash2, Edit3, Check, X, CheckSquare, Square, Info } from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";

/**
 * ModelLibrary — Trained models CRUD (rename, delete, set default).
 * Supports multi-select with batch delete.
 */
export default function ModelLibrary() {
  const { models, modelsLoading, fetchModels, renameModel, deleteModel, setDefaultModel, batchDeleteModels } = useSettingsStore();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [batchConfirm, setBatchConfirm] = useState(false);
  const [batchDeleting, setBatchDeleting] = useState(false);

  useEffect(() => { fetchModels(); }, [fetchModels]);

  const handleRename = async (id: string) => {
    if (editName.trim()) {
      await renameModel(id, editName.trim());
    }
    setEditingId(null);
  };

  const handleDelete = async (id: string) => {
    await deleteModel(id);
    setConfirmDeleteId(null);
    selectedIds.delete(id);
    setSelectedIds(new Set(selectedIds));
  };

  const toggleSelect = (id: string) => {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedIds(next);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === models.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(models.map((m) => m.id)));
    }
  };

  const handleBatchDelete = async () => {
    setBatchDeleting(true);
    try {
      await batchDeleteModels(Array.from(selectedIds));
      setSelectedIds(new Set());
    } catch {} finally {
      setBatchDeleting(false);
      setBatchConfirm(false);
    }
  };

  const targetColors: Record<string, string> = {
    d33: "var(--chart-d33)",
    tc: "var(--chart-tc)",
    vickers_hardness: "var(--chart-hardness)",
  };

  const targetLabels: Record<string, string> = {
    d33: "d₃₃",
    tc: "Tc",
    vickers_hardness: "Hardness",
  };

  const allSelected = models.length > 0 && selectedIds.size === models.length;
  const someSelected = selectedIds.size > 0;

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Brain size={18} /></div>
        <div>
          <h3>Trained Models Library</h3>
          <p className="settings-section-desc">
            Manage all trained models — rename, set defaults, select &amp; delete. UUID stays constant.
          </p>
        </div>
        <span className="settings-badge">{models.length}</span>
      </div>

      {modelsLoading ? (
        <div className="settings-loading">Loading models...</div>
      ) : models.length === 0 ? (
        <div className="settings-empty">No trained models yet. Train a model first.</div>
      ) : (
        <>
          {/* Batch actions toolbar */}
          <div className="model-batch-toolbar">
            <button className="model-select-all-btn" onClick={toggleSelectAll}
              title={allSelected ? "Deselect all" : "Select all"}>
              {allSelected ? <CheckSquare size={14} /> : <Square size={14} />}
              {allSelected ? "Deselect All" : "Select All"}
            </button>
            {someSelected && (
              <>
                <span className="model-selected-count">{selectedIds.size} selected</span>
                {batchConfirm ? (
                  <div className="model-batch-confirm">
                    <span>Delete {selectedIds.size} model{selectedIds.size > 1 ? "s" : ""}?</span>
                    <button className="model-lib-btn danger" onClick={handleBatchDelete} disabled={batchDeleting}>
                      {batchDeleting ? "Deleting..." : "Confirm"}
                    </button>
                    <button className="model-lib-btn" onClick={() => setBatchConfirm(false)}>Cancel</button>
                  </div>
                ) : (
                  <button className="model-batch-delete-btn" onClick={() => setBatchConfirm(true)}>
                    <Trash2 size={13} /> Delete Selected
                  </button>
                )}
              </>
            )}
          </div>

          <div className="model-lib-grid">
            {models.map((m) => (
              <div key={m.id} className={`model-lib-card ${m.is_default ? "is-default" : ""} ${selectedIds.has(m.id) ? "is-selected" : ""}`}>
                <div className="model-lib-header">
                  <button className="model-select-btn" onClick={() => toggleSelect(m.id)}
                    title={selectedIds.has(m.id) ? "Deselect" : "Select"}>
                    {selectedIds.has(m.id) ? <CheckSquare size={14} /> : <Square size={14} />}
                  </button>
                  <span
                    className="model-lib-target"
                    style={{ background: `${targetColors[m.target] || "var(--primary)"}22`, color: targetColors[m.target] }}
                  >
                    {targetLabels[m.target] || m.target}
                  </span>
                  <button
                    className={`model-lib-star ${m.is_default ? "active" : ""}`}
                    onClick={() => setDefaultModel(m.id)}
                    title={m.is_default ? "Default model" : "Set as default"}
                  >
                    <Star size={14} fill={m.is_default ? "currentColor" : "none"} />
                  </button>
                </div>

                {editingId === m.id ? (
                  <div className="model-lib-rename">
                    <input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleRename(m.id)}
                      autoFocus
                    />
                    <button onClick={() => handleRename(m.id)}><Check size={14} /></button>
                    <button onClick={() => setEditingId(null)}><X size={14} /></button>
                  </div>
                ) : (
                  <div className="model-lib-name" title={m.display_name}>
                    {m.display_name}
                  </div>
                )}

                <div className="model-lib-meta">
                  <span>{m.algorithm}</span>
                  <span>R²: {m.r2_score?.toFixed(3) ?? "—"}</span>
                  <span>RMSE: {m.rmse?.toFixed(2) ?? "—"}</span>
                </div>

                <div className="model-lib-details">
                  <span>Train: {m.n_train_samples} | Test: {m.n_test_samples}</span>
                  <span className="model-lib-uuid" title={m.id}>
                    UUID: {m.id.slice(0, 8)}…
                  </span>
                </div>

                <div className="model-lib-actions">
                  <button
                    className="model-lib-btn"
                    onClick={() => { setEditingId(m.id); setEditName(m.display_name); }}
                    title="Rename"
                  >
                    <Edit3 size={13} /> Rename
                  </button>
                  {confirmDeleteId === m.id ? (
                    <div className="model-lib-confirm">
                      <span>Delete?</span>
                      <button className="model-lib-btn danger" onClick={() => handleDelete(m.id)}>Yes</button>
                      <button className="model-lib-btn" onClick={() => setConfirmDeleteId(null)}>No</button>
                    </div>
                  ) : (
                    <button
                      className="model-lib-btn danger"
                      onClick={() => setConfirmDeleteId(m.id)}
                      title="Delete"
                    >
                      <Trash2 size={13} /> Delete
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
