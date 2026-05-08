"use client";

/**
 * Piezo.AI — Dataset Explorer
 * ==============================
 * Full-featured data table for viewing and editing materials.
 * Used as Step 4 of the wizard AND as standalone explorer for 'ready' datasets.
 *
 * Features: search, sort, pagination, inline CRUD, add/delete rows,
 * save/cancel bar, uid column, download CSV.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Search,
  Plus,
  Trash2,
  Download,
  Loader2,
  CheckCircle2,
  XCircle,
  Table2,
  X,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import {
  getMaterials,
  bulkUpdateMaterials,
  addMaterial,
  finalizeDataset,
  getDataset,
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

/* ---------- Parse status badge ---------- */

function StatusBadge({ status }: { status: string }) {
  const configs: Record<string, { icon: typeof CheckCircle2; cls: string }> = {
    success: { icon: CheckCircle2, cls: "success" },
    error: { icon: XCircle, cls: "error" },
    unsupported_elements: { icon: XCircle, cls: "warning" },
    pending: { icon: Loader2, cls: "muted" },
  };
  const cfg = configs[status] || configs.pending;
  const Icon = cfg.icon;
  return (
    <span className={`status-badge-inline ${cfg.cls}`} title={status}>
      <Icon size={14} />
    </span>
  );
}

/* ---------- Component ---------- */

export default function DatasetExplorer() {
  const {
    activeDatasetId,
    activeDataset,
    materials,
    totalMaterials,
    currentPage,
    pageSize,
    totalPages,
    isLoadingMaterials,
    searchQuery,
    sortBy,
    sortOrder,
    selectedIds,
    editedCells,
    pendingDeletes,
    newRows,
    lastSelectedId,
    setMaterials,
    setIsLoadingMaterials,
    setSearchQuery,
    setSorting,
    setCurrentPage,
    setPageSize,
    toggleRowSelection,
    deselectAll,
    setLastSelectedId,
    editCell,
    markForDeletion,
    addNewRow,
    discardChanges,
    setActiveDataset,
    enterWizardAtStep,
  } = useDatasetStore();

  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveErrorDetails, setSaveErrorDetails] = useState<string[]>([]);
  const [isFinalized, setIsFinalized] = useState(false);
  const [searchInput, setSearchInput] = useState(searchQuery);
  /* Reactive change detection — computed from store state */
  const hasChanges = useMemo(
    () => editedCells.size > 0 || pendingDeletes.size > 0 || newRows.length > 0,
    [editedCells, pendingDeletes, newRows],
  );
  const changesSummary = useMemo(() => {
    const parts: string[] = [];
    if (editedCells.size > 0) parts.push(`${editedCells.size} edit${editedCells.size > 1 ? "s" : ""}`);
    if (newRows.length > 0) parts.push(`${newRows.length} new row${newRows.length > 1 ? "s" : ""}`);
    if (pendingDeletes.size > 0) parts.push(`${pendingDeletes.size} deletion${pendingDeletes.size > 1 ? "s" : ""}`);
    return parts.join(", ");
  }, [editedCells, newRows, pendingDeletes]);

  /* Load materials */
  const loadMaterials = useCallback(async () => {
    if (!activeDatasetId) return;
    setIsLoadingMaterials(true);
    try {
      const result = await getMaterials(activeDatasetId, {
        search: searchQuery || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: currentPage,
        page_size: pageSize,
      });
      setMaterials(result.items, result.total, result.page, result.total_pages);
    } catch {
      /* silently fail — empty table */
    } finally {
      setIsLoadingMaterials(false);
    }
  }, [activeDatasetId, searchQuery, sortBy, sortOrder, currentPage, pageSize, setMaterials, setIsLoadingMaterials]);

  useEffect(() => {
    loadMaterials();
  }, [loadMaterials]);

  /* Debounced search */
  useEffect(() => {
    const timeout = setTimeout(() => {
      setSearchQuery(searchInput);
    }, 300);
    return () => clearTimeout(timeout);
  }, [searchInput, setSearchQuery]);

  /* Table columns — show ALL mapped fields as editable */
  const tableColumns: ColumnDef[] = useMemo(() => {
    const cols: ColumnDef[] = [
      { key: "uid", label: "UID", width: 60, type: "int", sortable: true, sticky: true },
      { key: "formula", label: "Formula", width: 220, type: "string", editable: true, sortable: true },
      { key: "d33", label: "d₃₃", width: 80, type: "float", editable: true, sortable: true },
      { key: "tc", label: "Tc", width: 80, type: "float", editable: true, sortable: true },
      { key: "vickers_hardness", label: "Hardness", width: 90, type: "float", editable: true, sortable: true },
    ];

    /* Add sintering/fabrication if any row has them */
    if (materials.some((m) => m.sintering_temp_c != null)) {
      cols.push({ key: "sintering_temp_c", label: "Sint.Temp", width: 90, type: "float", editable: true, sortable: true });
    }
    if (materials.some((m) => m.fabrication_method != null)) {
      cols.push({ key: "fabrication_method", label: "Fabrication", width: 110, type: "string", editable: true });
    }

    /* Composite columns if dataset has them */
    if (activeDataset?.has_composite_fields) {
      cols.push(
        { key: "matrix_type", label: "Matrix", width: 100, type: "string", editable: true },
        { key: "filler_wt_pct", label: "Filler%", width: 80, type: "float", editable: true, sortable: true },
        { key: "particle_morphology", label: "Morphology", width: 110, type: "string", editable: true },
        { key: "particle_size_nm", label: "Size(nm)", width: 80, type: "float", editable: true },
        { key: "surface_treatment", label: "Treatment", width: 100, type: "string", editable: true },
      );
    }

    /* Extra numeric cols if mapped */
    if (materials.some((m) => m.qm != null)) {
      cols.push({ key: "qm", label: "Qm", width: 80, type: "float", editable: true, sortable: true });
    }
    if (materials.some((m) => m.kp != null)) {
      cols.push({ key: "kp", label: "kp", width: 80, type: "float", editable: true, sortable: true });
    }

    /* Parse status (always last, read-only) */
    cols.push({
      key: "parse_status",
      label: "Status",
      width: 70,
      render: (v) => <StatusBadge status={String(v)} />,
    });

    return cols;
  }, [activeDataset, materials]);

  /* Combine DB materials + new rows */
  const allData = useMemo(() => {
    const dbRows = materials.map((m) => ({ ...m } as Record<string, unknown>));
    const tempRows = newRows.map((r) => ({ ...r, _isNew: true } as Record<string, unknown>));
    return [...dbRows, ...tempRows];
  }, [materials, newRows]);

  /* Selection handlers */
  const handleSelectionChange = useCallback(
    (ids: Set<string>) => {
      deselectAll();
      ids.forEach((id) => toggleRowSelection(id));
    },
    [deselectAll, toggleRowSelection],
  );

  /* Add row */
  const handleAddRow = useCallback(() => {
    const maxUid = Math.max(
      ...materials.map((m) => m.uid),
      ...newRows.map((r) => r.uid),
      0,
    );
    const tempRow: MaterialRow = {
      id: `__new_${Date.now()}`,
      uid: maxUid + 1,
      formula: "",
      d33: null, tc: null, vickers_hardness: null, qm: null, kp: null,
      relative_density_pct: null, sintering_temp_c: null,
      sintering_method: null, ceramic_type: null, fabrication_method: null,
      matrix_type: "none", filler_wt_pct: 0, particle_morphology: "none",
      particle_size_nm: null, surface_treatment: "none",
      source_doi: null, source_notes: null,
      parse_status: "pending", parse_warnings: null,
    };
    addNewRow(tempRow);
  }, [materials, newRows, addNewRow]);

  /* Delete selected */
  const handleDeleteSelected = useCallback(() => {
    markForDeletion(Array.from(selectedIds));
    deselectAll();
  }, [selectedIds, markForDeletion, deselectAll]);

  /* Save all changes */
  const handleSave = useCallback(async () => {
    if (!activeDatasetId) return;
    setIsSaving(true);
    setSaveError(null);
    setSaveErrorDetails([]);

    try {
      const state = useDatasetStore.getState();
      const errors: string[] = [];

      // Save new rows first
      for (const row of state.newRows) {
        try {
          await addMaterial(activeDatasetId, {
            formula: row.formula || "EMPTY",
            d33: row.d33, tc: row.tc, vickers_hardness: row.vickers_hardness,
            matrix_type: row.matrix_type, filler_wt_pct: row.filler_wt_pct,
            particle_morphology: row.particle_morphology,
            particle_size_nm: row.particle_size_nm,
            surface_treatment: row.surface_treatment,
          });
        } catch (err) {
          errors.push(err instanceof Error ? err.message : "Failed to add new row");
        }
      }

      // Bulk update/delete existing
      const updates = Array.from(state.editedCells.entries()).map(
        ([id, changes]) => ({ id, updates: changes as Record<string, unknown> }),
      );
      const deletes = Array.from(state.pendingDeletes);

      if (updates.length > 0 || deletes.length > 0) {
        const result = await bulkUpdateMaterials(activeDatasetId, updates, deletes);
        if (result.errors?.length) {
          errors.push(...withUidErrorDetails(result.errors, materials));
        }
      }

      // Always reset local edits then refresh from DB truth (invalid edits are effectively reverted)
      discardChanges();
      await loadMaterials();

      // Refresh dataset detail (status may flip to pending after edits)
      const ds = await getDataset(activeDatasetId);
      setActiveDataset(ds);

      if (errors.length > 0) {
        setSaveError("Some changes were not saved. Invalid edits were reverted to previous values.");
        setSaveErrorDetails(errors);
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
      setSaveErrorDetails([]);
    } finally {
      setIsSaving(false);
    }
  }, [activeDatasetId, discardChanges, loadMaterials, setActiveDataset]);

  /* Finalize (mark as ready) */
  const handleFinalize = useCallback(async () => {
    if (!activeDatasetId) return;
    try {
      const ds = await finalizeDataset(activeDatasetId);
      setActiveDataset(ds);
      setIsFinalized(true);
    } catch { /* ignore */ }
  }, [activeDatasetId, setActiveDataset]);

  /* Download CSV */
  const handleDownload = useCallback(() => {
    if (materials.length === 0) return;
    const headers = tableColumns.map((c) => c.key);
    const csvRows = [
      headers.join(","),
      ...materials.map((m) =>
        headers.map((h) => {
          const v = (m as Record<string, unknown>)[h];
          if (v == null) return "";
          const s = String(v);
          return s.includes(",") ? `"${s}"` : s;
        }).join(","),
      ),
    ];
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeDataset?.display_name || "dataset"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [materials, tableColumns, activeDataset]);

  /* Row highlight map */
  const highlightedRows = useMemo(() => {
    const map = new Map<string, "success" | "warning" | "error">();
    materials.forEach((m) => {
      if (m.parse_status === "error") map.set(m.id, "error");
      else if (m.parse_status === "unsupported_elements") map.set(m.id, "warning");
    });
    return map;
  }, [materials]);

  return (
    <div className="explorer-container">
      {activeDataset?.status === "pending" && (
        <NoticeBanner
          variant="warning"
          title="Dataset needs re-validation"
          message="You edited this dataset after it was marked ready. Re-run Review Issues, then mark it ready again."
          compact
          action={(
            <button className="btn-ghost btn-sm" onClick={() => enterWizardAtStep("review")}>
              Re-run Review Issues
            </button>
          )}
        />
      )}
      {/* Header */}
      {saveError && (
        <NoticeBanner
          variant="error"
          title="Save issues detected"
          message={saveError}
          details={saveErrorDetails}
          onDismiss={() => {
            setSaveError(null);
            setSaveErrorDetails([]);
          }}
        />
      )}

      <div className="explorer-header">
        <div className="explorer-header-left">
          <Table2 size={18} />
          <h3>{activeDataset?.display_name || "Dataset"}</h3>
          {activeDataset && (
            <span className={`explorer-status-badge ${activeDataset.status}`}>
              {activeDataset.status}
            </span>
          )}
          <span className="explorer-stats">{totalMaterials} rows</span>
        </div>

        <div className="explorer-toolbar">
          <div className="explorer-search">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search all fields..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
            {searchInput && (
              <button onClick={() => setSearchInput("")} style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: 2 }}>
                <X size={14} />
              </button>
            )}
          </div>
          <button className="btn-ghost btn-sm" onClick={handleAddRow} title="Add row">
            <Plus size={14} /> Add Row
          </button>
          {selectedIds.size > 0 && (
            <button className="btn-ghost btn-sm btn-danger-text" onClick={handleDeleteSelected}>
              <Trash2 size={14} /> Delete ({selectedIds.size})
            </button>
          )}
          <button className="btn-ghost btn-sm" onClick={handleDownload} title="Download CSV">
            <Download size={14} />
          </button>
        </div>
      </div>

      {/* Loading */}
      {isLoadingMaterials && materials.length === 0 && (
        <div className="review-loading">
          <Loader2 size={24} className="spin" />
          <p>Loading materials...</p>
        </div>
      )}

      {/* Table — pagination handled by DataTable */}
      {(!isLoadingMaterials || materials.length > 0) && (
        <DataTable
          columns={tableColumns}
          data={allData}
          selectable
          selectedIds={selectedIds}
          onSelectionChange={handleSelectionChange}
          lastSelectedId={lastSelectedId}
          onLastSelectedChange={setLastSelectedId}
          editable
          editedCells={editedCells}
          onCellEdit={editCell}
          sortBy={sortBy}
          sortOrder={sortOrder}
          onSort={setSorting}
          onDeleteRow={(id) => markForDeletion([id])}
          highlightedRows={highlightedRows}
          deletedIds={pendingDeletes}
          paginate
          externalPage={currentPage}
          externalTotalPages={totalPages}
          externalTotal={totalMaterials}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
          defaultPageSize={pageSize}
          externalPageSize={pageSize}
          pageSizeOptions={[25, 50, 100]}
          maxHeight="calc(100vh - 380px)"
          emptyMessage="No materials found"
        />
      )}

      {/* Save bar — reactive */}
      {hasChanges && (
        <div className="save-bar fixed-bottom">
          <span className="save-bar-summary">{changesSummary}</span>
          <div className="save-bar-actions">
            <button className="btn-ghost" onClick={discardChanges}>Cancel</button>
            <button className="btn-primary" onClick={handleSave} disabled={isSaving}>
              {isSaving ? <><Loader2 size={14} className="spin" /> Saving...</> : "Save All Changes"}
            </button>
          </div>
        </div>
      )}

      {/* Finalize */}
      {activeDataset?.status === "pending" && !hasChanges && !isFinalized && (
        <div className="explorer-finalize">
          <button className="btn-primary" onClick={handleFinalize}>
            <CheckCircle2 size={16} />
            Mark Dataset as Ready
          </button>
        </div>
      )}

      {isFinalized && (
        <div className="explorer-finalized">
          <CheckCircle2 size={18} />
          <span>Dataset marked as ready! You can now use it for training.</span>
        </div>
      )}
    </div>
  );
}
