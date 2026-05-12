"use client";

/**
 * Piezo.AI — Review Issues Step
 * ================================
 * Data quality review with issue detection, inline editing,
 * multi-select with shift-click, and bulk actions.
 * Step 3 of the dataset upload wizard.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Trash2,
  PartyPopper,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import {
  getQualityReport,
  bulkUpdateMaterials,
  clearMaterialColumn,
  getMaterials,
  type MaterialRow,
} from "@/lib/api/datasets";
import DataTable, { type ColumnDef } from "./DataTable";
import NoticeBanner from "@/components/common/NoticeBanner";

function withUidErrorDetails(
  errors: string[],
  rows: MaterialRow[],
): string[] {
  const idToUid = new Map(rows.map((r) => [r.id, r.uid]));
  return errors.map((err) => {
    const match = err.match(/^(Update|Delete)\s+([0-9a-fA-F-]{36})(:.*)$/);
    if (!match) return err;
    const [, op, rowId, rest] = match;
    const uid = idToUid.get(rowId);
    if (uid == null) return `${op} ${rowId}${rest}`;
    return `${op} uid=${uid} (${rowId})${rest}`;
  });
}

/* ---------- Component ---------- */

export default function ReviewIssuesStep() {
  const {
    activeDatasetId,
    qualityReport,
    isLoadingReport,
    selectedIds,
    editedCells,
    pendingDeletes,
    lastSelectedId,
    setQualityReport,
    setIsLoadingReport,
    toggleRowSelection,
    deselectAll,
    setLastSelectedId,
    editCell,
    markForDeletion,
    discardChanges,
    setWizardStep,
  } = useDatasetStore();

  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveErrorDetails, setSaveErrorDetails] = useState<string[]>([]);
  const [materials, setMaterials] = useState<MaterialRow[]>([]);
  const [selectedColumnsForDelete, setSelectedColumnsForDelete] = useState<Set<string>>(new Set());

  /* Reactive change detection */
  const hasChanges = useMemo(
    () => editedCells.size > 0 || pendingDeletes.size > 0,
    [editedCells, pendingDeletes],
  );
  const changesSummary = useMemo(() => {
    const parts: string[] = [];
    if (editedCells.size > 0) parts.push(`${editedCells.size} edit${editedCells.size > 1 ? "s" : ""}`);
    if (pendingDeletes.size > 0) parts.push(`${pendingDeletes.size} deletion${pendingDeletes.size > 1 ? "s" : ""}`);
    return parts.join(", ");
  }, [editedCells, pendingDeletes]);

  /* Load quality report — re-fetch when activeDatasetId changes */
  useEffect(() => {
    if (!activeDatasetId) return;
    setIsLoadingReport(true);
    getQualityReport(activeDatasetId)
      .then(setQualityReport)
      .catch(() => setQualityReport(null))
      .finally(() => setIsLoadingReport(false));
  }, [activeDatasetId, setQualityReport, setIsLoadingReport]);

  useEffect(() => {
    if (!activeDatasetId) return;
    getMaterials(activeDatasetId, { page: 1, page_size: 5000, sort_by: "uid", sort_order: "asc" })
      .then((res) => setMaterials(res.items))
      .catch(() => setMaterials([]));
  }, [activeDatasetId]);

  const issueByMaterial = useMemo(() => {
    const map = new Map<string, { columns: Set<string>; messages: string[] }>();
    if (!qualityReport) return map;
    qualityReport.issues.forEach((issue) => {
      const existing = map.get(issue.material_id) ?? { columns: new Set<string>(), messages: [] };
      existing.columns.add(issue.column);
      existing.messages.push(`${issue.column}: ${issue.message}`);
      map.set(issue.material_id, existing);
    });
    return map;
  }, [qualityReport]);

  const highlightedRows = useMemo(() => {
    const map = new Map<string, "success" | "warning" | "error">();
    issueByMaterial.forEach((_v, materialId) => map.set(materialId, "error"));
    return map;
  }, [issueByMaterial]);

  const tableColumns: ColumnDef[] = useMemo(() => {
    const cols: ColumnDef[] = [
      { key: "uid", label: "UID", width: 60, type: "int", sortable: true, sticky: true },
      { key: "formula", label: "Formula", width: 220, type: "string", editable: true, sortable: true },
      { key: "d33", label: "d₃₃", width: 80, type: "float", editable: true, sortable: true },
      { key: "tc", label: "Tc", width: 80, type: "float", editable: true, sortable: true },
      { key: "vickers_hardness", label: "Hardness", width: 100, type: "float", editable: true, sortable: true },
      { key: "sintering_temp_c", label: "Sint.Temp", width: 90, type: "float", editable: true, sortable: true },
      { key: "fabrication_method", label: "Fabrication", width: 110, type: "string", editable: true },
      { key: "matrix_type", label: "Matrix", width: 100, type: "string", editable: true },
      { key: "filler_wt_pct", label: "Filler%", width: 80, type: "float", editable: true, sortable: true },
      { key: "particle_morphology", label: "Morphology", width: 110, type: "string", editable: true },
      { key: "particle_size_nm", label: "Size(nm)", width: 80, type: "float", editable: true },
      { key: "surface_treatment", label: "Treatment", width: 100, type: "string", editable: true },
      { key: "qm", label: "Qm", width: 80, type: "float", editable: true, sortable: true },
      { key: "kp", label: "kp", width: 80, type: "float", editable: true, sortable: true },
      {
        key: "__issue_reason",
        label: "Issue Reason",
        width: 420,
        type: "string",
        render: (_v, row) => {
          const materialId = String(row.id ?? "");
          const issue = issueByMaterial.get(materialId);
          if (!issue || issue.messages.length === 0) return "—";
          return issue.messages.join(" | ");
        },
      },
    ];
    return cols;
  }, [issueByMaterial]);

  const materialData = useMemo(
    () => materials.map((m) => ({ ...m } as Record<string, unknown>)),
    [materials],
  );

  /* Selection handlers */
  const handleSelectionChange = useCallback(
    (ids: Set<string>) => {
      deselectAll();
      ids.forEach((id) => toggleRowSelection(id));
    },
    [deselectAll, toggleRowSelection],
  );

  /* Delete selected issues' materials */
  const handleDeleteSelected = useCallback(() => {
    markForDeletion(Array.from(selectedIds));
    deselectAll();
  }, [selectedIds, markForDeletion, deselectAll]);

  /* Save changes */
  const handleSave = useCallback(async () => {
    if (!activeDatasetId) return;
    setIsSaving(true);
    setSaveError(null);
    setSaveErrorDetails([]);

    try {
      const state = useDatasetStore.getState();
      const updates = Array.from(state.editedCells.entries()).map(
        ([id, changes]) => ({ id, updates: changes as Record<string, unknown> }),
      );
      const deletes = Array.from(state.pendingDeletes);

      const result = await bulkUpdateMaterials(activeDatasetId, updates, deletes);
      discardChanges();

      // Reload quality report
      const report = await getQualityReport(activeDatasetId);
      setQualityReport(report);
      if (result.errors?.length) {
        setSaveError("Some edits were rejected and reverted.");
        setSaveErrorDetails(withUidErrorDetails(result.errors, materials));
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
      setSaveErrorDetails([]);
    } finally {
      setIsSaving(false);
    }
  }, [activeDatasetId, discardChanges, materials, setQualityReport]);

  const targetColumnsPresent = useMemo(() => {
    const present = new Set<string>();
    ["d33", "tc", "vickers_hardness"].forEach((k) => {
      if (materials.some((m) => (m as unknown as Record<string, unknown>)[k] != null)) present.add(k);
    });
    return present;
  }, [materials]);

  const columnOptions = useMemo(() => {
    const base = tableColumns.map((c) => c.key);
    return base.filter((k) => !["uid", "formula"].includes(k));
  }, [tableColumns]);

  const toggleColumnSelection = useCallback((col: string) => {
    setSelectedColumnsForDelete((prev) => {
      const next = new Set(prev);
      if (next.has(col)) next.delete(col);
      else next.add(col);
      return next;
    });
  }, []);

  const handleSelectAllColumns = useCallback(() => {
    // by default keep d33 unselected to maintain at least one core target
    setSelectedColumnsForDelete(new Set(columnOptions.filter((c) => c !== "d33")));
  }, [columnOptions]);

  const handleClearSelectedColumns = useCallback(async () => {
    if (!activeDatasetId || selectedColumnsForDelete.size === 0) return;
    setIsSaving(true);
    setSaveError(null);
    setSaveErrorDetails([]);
    try {
      // must retain at least one core target among currently present
      const deletingCore = [...targetColumnsPresent].filter((c) => selectedColumnsForDelete.has(c));
      if (deletingCore.length === targetColumnsPresent.size && targetColumnsPresent.size > 0) {
        setSaveError("At least one core target must remain: d33 or tc or hardness.");
        setSaveErrorDetails([
          "You selected all available target columns for deletion.",
          "Keep at least one among d33/tc/vickers_hardness to preserve dataset utility.",
        ]);
        return;
      }

      const allErrors: string[] = [];
      for (const col of selectedColumnsForDelete) {
        const res = await clearMaterialColumn(activeDatasetId, col);
        if (res.errors?.length) allErrors.push(...res.errors);
      }
      if (allErrors.length > 0) {
        setSaveError("Column clear completed with issues.");
        setSaveErrorDetails(allErrors);
      }
      const report = await getQualityReport(activeDatasetId);
      setQualityReport(report);
      setSelectedColumnsForDelete(new Set());
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Column clear failed");
      setSaveErrorDetails([]);
    } finally {
      setIsSaving(false);
    }
  }, [activeDatasetId, selectedColumnsForDelete, setQualityReport, targetColumnsPresent]);

  /* Continue to explorer */
  const handleContinue = useCallback(() => {
    setWizardStep("explore");
  }, [setWizardStep]);

  /* Handle cell edit on material row */
  const handleCellEdit = useCallback(
    (rowId: string, field: string, value: unknown) => {
      editCell(rowId, field, value);
    },
    [editCell],
  );

  /* Loading state */
  if (isLoadingReport) {
    return (
      <div className="review-loading">
        <Loader2 size={24} className="spin" />
        <p>Analyzing dataset quality...</p>
      </div>
    );
  }

  /* No issues */
  if (qualityReport && qualityReport.issue_count === 0) {
    return (
      <div className="review-success">
        <div className="review-success-icon">
          <PartyPopper size={40} />
        </div>
        <h3>Your dataset looks great!</h3>
        <p>No issues detected. All {qualityReport.total_rows} rows are valid.</p>
        <button className="btn-primary" onClick={handleContinue}>
          Continue to Explorer →
        </button>
      </div>
    );
  }

  if (!qualityReport) {
    return (
      <div className="review-loading">
        <AlertTriangle size={24} />
        <p>Quality report unavailable. Please go back and re-apply mapping.</p>
      </div>
    );
  }

  return (
    <div className="review-step">
      {/* Summary stats */}
      <div className="quality-summary">
        <div className="quality-stat-card">
          <div className="quality-stat-value">{qualityReport.total_rows}</div>
          <div className="quality-stat-label">Total Rows</div>
        </div>
        <div className="quality-stat-card success">
          <CheckCircle2 size={16} />
          <div className="quality-stat-value">{qualityReport.valid_rows}</div>
          <div className="quality-stat-label">Valid</div>
        </div>
        <div className="quality-stat-card warning">
          <AlertTriangle size={16} />
          <div className="quality-stat-value">{qualityReport.issue_count}</div>
          <div className="quality-stat-label">Issues</div>
        </div>
        <div className="quality-stat-card">
          <div className="quality-stat-value">{qualityReport.column_stats.length}</div>
          <div className="quality-stat-label">Columns Checked</div>
        </div>
      </div>

      {/* Column breakdown */}
      <div className="quality-column-breakdown">
        <h4>Column Quality</h4>
        <div className="quality-columns-grid">
          {qualityReport.column_stats
            .filter((cs) => cs.missing_count > 0 || cs.invalid_count > 0)
            .map((cs) => {
              const validPct = Math.round(
                ((cs.total_values - cs.missing_count - cs.invalid_count) /
                  cs.total_values) *
                  100,
              );
              return (
                <div key={cs.column_name} className="quality-column-item">
                  <div className="quality-column-header">
                    <span className="quality-column-name">{cs.column_name}</span>
                    <span className="quality-column-pct">{validPct}% valid</span>
                  </div>
                  <div className="quality-progress-bar">
                    <div
                      className="quality-progress-fill"
                      style={{ width: `${validPct}%` }}
                    />
                  </div>
                  <div className="quality-column-counts">
                    {cs.missing_count > 0 && (
                      <span className="quality-count missing">
                        {cs.missing_count} missing
                      </span>
                    )}
                    {cs.invalid_count > 0 && (
                      <span className="quality-count invalid">
                        {cs.invalid_count} invalid
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Bulk actions bar */}
      {selectedIds.size > 0 && (
        <div className="bulk-actions-bar">
          <span className="bulk-actions-count">
            {selectedIds.size} row{selectedIds.size > 1 ? "s" : ""} selected
          </span>
          <div className="bulk-actions-buttons">
            <button className="btn-ghost btn-sm" onClick={deselectAll}>
              Deselect All
            </button>
            <button className="btn-ghost btn-sm btn-danger-text" onClick={handleDeleteSelected}>
              <Trash2 size={14} />
              Delete Selected Rows
            </button>
          </div>
        </div>
      )}

      {/* Quick select: all rows with issues */}
      {qualityReport && qualityReport.issue_count > 0 && (
        <div className="bulk-actions-bar" style={{ marginTop: 4 }}>
          <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
            {qualityReport.issue_count} rows with issues
          </span>
          <button
            className="btn-ghost btn-sm"
            onClick={() => {
              deselectAll();
              materials.forEach((m) => {
                const materialId = String(m.id ?? "");
                if (issueByMaterial.has(materialId)) {
                  toggleRowSelection(materialId);
                }
              });
            }}
          >
            Select All with Issues
          </button>
        </div>
      )}

      {/* Column remediation */}
      <div className="review-actions" style={{ justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 8, width: "100%" }}>
          <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>Columns to clear across all rows:</span>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button className="btn-ghost btn-sm" onClick={handleSelectAllColumns}>Select All (keep d33 by default)</button>
            <button className="btn-ghost btn-sm" onClick={() => setSelectedColumnsForDelete(new Set())}>Deselect All</button>
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {columnOptions.map((col) => (
              <label key={col} className="filter-dropdown-item" style={{ border: "1px solid var(--border)", borderRadius: 6, padding: "6px 10px" }}>
                <input
                  type="checkbox"
                  checked={selectedColumnsForDelete.has(col)}
                  onChange={() => toggleColumnSelection(col)}
                />
                {col}
              </label>
            ))}
          </div>
          <button className="btn-ghost btn-sm btn-danger-text" disabled={selectedColumnsForDelete.size === 0 || isSaving} onClick={handleClearSelectedColumns}>
            Clear Selected Columns ({selectedColumnsForDelete.size})
          </button>
        </div>
      </div>

      {/* Full review table */}
      <div className="review-issues-table">
        <h4>Review Table ({materials.length} rows, {qualityReport.issue_count} issues marked)</h4>
        <DataTable
          columns={tableColumns}
          data={materialData}
          selectable
          selectedIds={selectedIds}
          onSelectionChange={handleSelectionChange}
          lastSelectedId={lastSelectedId}
          onLastSelectedChange={setLastSelectedId}
          editable
          editedCells={editedCells}
          onCellEdit={handleCellEdit}
          onDeleteRow={(id) => markForDeletion([id])}
          deletedIds={pendingDeletes}
          highlightedRows={highlightedRows}
          paginate
          defaultPageSize={25}
          pageSizeOptions={[10, 25, 50]}
          maxHeight="600px"
          emptyMessage="No rows found"
        />
      </div>

      {/* Save error */}
      {saveError && (
        <NoticeBanner
          variant="error"
          title="Validation issues"
          message={saveError}
          details={saveErrorDetails}
          onDismiss={() => {
            setSaveError(null);
            setSaveErrorDetails([]);
          }}
        />
      )}

      {/* Save bar */}
      <div className="review-actions">
        {hasChanges && (
          <div className="save-bar">
            <span className="save-bar-summary">{changesSummary}</span>
            <div className="save-bar-actions">
              <button className="btn-ghost" onClick={discardChanges}>
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={handleSave}
                disabled={isSaving}
              >
                {isSaving ? (
                  <>
                    <Loader2 size={14} className="spin" />
                    Saving...
                  </>
                ) : (
                  "Save Changes"
                )}
              </button>
            </div>
          </div>
        )}
        {!hasChanges && (
          <button className="btn-primary" onClick={handleContinue}>
            Continue to Explorer →
          </button>
        )}
      </div>
    </div>
  );
}
