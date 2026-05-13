"use client";

/**
 * Piezo.AI — Reusable DataTable Component
 * =========================================
 * Regular DOM table with sorting, selection (shift-select),
 * inline editing, sticky header, and built-in client-side pagination.
 *
 * Used by: ReviewIssuesStep, DatasetExplorer
 *
 * NOTE: Virtualizer removed — pagination (25/50/100 per page) handles
 * large datasets without performance issues. This also fixes the row
 * overlapping bug that occurred with absolute positioning + text wrapping.
 */

import {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
  type KeyboardEvent,
} from "react";
import { ArrowUp, ArrowDown, Pencil, Trash2 } from "lucide-react";

/* ---------- Types ---------- */

export interface ColumnDef {
  key: string;
  label: string;
  width?: number;
  minWidth?: number;
  type?: "string" | "float" | "int" | "badge" | "checkbox";
  editable?: boolean;
  sortable?: boolean;
  sticky?: boolean;
  options?: string[]; // For dropdown editing
  render?: (value: unknown, row: Record<string, unknown>) => ReactNode;
}

interface DataTableProps {
  columns: ColumnDef[];
  data: Record<string, unknown>[];
  rowIdKey?: string;

  /* Selection */
  selectable?: boolean;
  selectedIds?: Set<string>;
  onSelectionChange?: (ids: Set<string>) => void;
  lastSelectedId?: string | null;
  onLastSelectedChange?: (id: string | null) => void;

  /* Editing */
  editable?: boolean;
  editedCells?: Map<string, Record<string, unknown>>;
  onCellEdit?: (rowId: string, field: string, value: unknown) => void;

  /* Sorting */
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  onSort?: (col: string, order: "asc" | "desc") => void;

  /* Actions */
  onDeleteRow?: (rowId: string) => void;

  /* Row highlighting */
  highlightedRows?: Map<string, "success" | "warning" | "error">;
  deletedIds?: Set<string>;

  /* Pagination — built-in client-side */
  paginate?: boolean;
  defaultPageSize?: number;
  pageSizeOptions?: number[];

  /* External pagination (server-side) — disables built-in */
  externalPage?: number;
  externalPageSize?: number;
  externalTotalPages?: number;
  externalTotal?: number;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (size: number) => void;

  /* Layout */
  maxHeight?: string;

  /* Empty state */
  emptyMessage?: string;
  emptyIcon?: ReactNode;
}

/* ---------- Component ---------- */

export default function DataTable({
  columns,
  data,
  rowIdKey = "id",
  selectable = false,
  selectedIds,
  onSelectionChange,
  lastSelectedId,
  onLastSelectedChange,
  editable = false,
  editedCells,
  onCellEdit,
  sortBy,
  sortOrder,
  onSort,
  onDeleteRow,
  highlightedRows,
  deletedIds,
  paginate = true,
  defaultPageSize = 50,
  pageSizeOptions = [25, 50, 100],
  externalPage,
  externalPageSize,
  externalTotalPages,
  externalTotal,
  onPageChange,
  onPageSizeChange,
  maxHeight = "calc(100vh - 320px)",
  emptyMessage = "No data to display",
  emptyIcon,
}: DataTableProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [editingCell, setEditingCell] = useState<{
    rowId: string;
    field: string;
  } | null>(null);
  const [editValue, setEditValue] = useState<string>("");
  const [editingRowId, setEditingRowId] = useState<string | null>(null);

  /* Built-in pagination state */
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    setPageSize(defaultPageSize);
  }, [defaultPageSize]);

  /* Determine if using external or built-in pagination */
  const useExternal = externalPage != null && onPageChange != null;

  /* Paginated data (built-in only) */
  const paginatedData = useMemo(() => {
    if (!paginate || useExternal) return data;
    const start = (currentPage - 1) * pageSize;
    return data.slice(start, start + pageSize);
  }, [data, paginate, useExternal, currentPage, pageSize]);

  const totalPages = useMemo(() => {
    if (useExternal) return externalTotalPages || 1;
    if (!paginate) return 1;
    return Math.max(1, Math.ceil(data.length / pageSize));
  }, [useExternal, externalTotalPages, paginate, data.length, pageSize]);

  const activePage = useExternal ? externalPage! : currentPage;
  const totalRows = useExternal ? (externalTotal || data.length) : data.length;
  const effectivePageSize = useExternal ? (externalPageSize ?? pageSize) : pageSize;
  const pageStart = totalRows === 0 ? 0 : (activePage - 1) * effectivePageSize + 1;
  const pageEnd = Math.min(totalRows, activePage * effectivePageSize);

  /* Unified page-size handler — works for both built-in and external pagination */
  const handlePageSizeChange = useCallback((size: number) => {
    setPageSize(size);
    setCurrentPage(1);
    if (useExternal && onPageSizeChange) {
      onPageSizeChange(size);
    }
  }, [useExternal, onPageSizeChange]);

  const goToPage = useCallback((page: number) => {
    if (useExternal) {
      onPageChange!(page);
    } else {
      setCurrentPage(page);
      scrollRef.current?.scrollTo({ top: 0 });
    }
  }, [useExternal, onPageChange]);

  /* Selection helpers */
  const displayData = useExternal ? data : paginatedData;

  const allSelected = useMemo(() => {
    if (!selectedIds || displayData.length === 0) return false;
    return displayData.every((r) => selectedIds.has(String(r[rowIdKey])));
  }, [selectedIds, displayData, rowIdKey]);

  const handleSelectAll = useCallback(() => {
    if (!onSelectionChange) return;
    if (allSelected) {
      onSelectionChange(new Set());
    } else {
      onSelectionChange(new Set(displayData.map((r) => String(r[rowIdKey]))));
    }
  }, [allSelected, displayData, rowIdKey, onSelectionChange]);

  const handleRowSelect = useCallback(
    (rowId: string, shiftKey: boolean) => {
      if (!onSelectionChange || !selectedIds) return;

      if (shiftKey && lastSelectedId) {
        const ids = displayData.map((r) => String(r[rowIdKey]));
        const startIdx = ids.indexOf(lastSelectedId);
        const endIdx = ids.indexOf(rowId);
        if (startIdx !== -1 && endIdx !== -1) {
          const [from, to] = startIdx < endIdx
            ? [startIdx, endIdx]
            : [endIdx, startIdx];
          const next = new Set(selectedIds);
          for (let i = from; i <= to; i++) next.add(ids[i]);
          onSelectionChange(next);
        }
      } else {
        const next = new Set(selectedIds);
        if (next.has(rowId)) next.delete(rowId);
        else next.add(rowId);
        onSelectionChange(next);
      }
      onLastSelectedChange?.(rowId);
    },
    [displayData, rowIdKey, selectedIds, lastSelectedId, onSelectionChange, onLastSelectedChange],
  );

  /* Inline editing */
  const startEditing = useCallback(
    (rowId: string, field: string, currentValue: unknown) => {
      if (!editable) return;
      setEditingCell({ rowId, field });
      setEditValue(currentValue == null ? "" : String(currentValue));
    },
    [editable],
  );

  const commitEdit = useCallback(() => {
    if (!editingCell || !onCellEdit) return;
    const col = columns.find((c) => c.key === editingCell.field);
    let finalValue: unknown = editValue;

    if (col?.type === "float" || col?.type === "int") {
      const num = Number(editValue);
      finalValue = editValue === "" ? null : isNaN(num) ? null : num;
    }

    onCellEdit(editingCell.rowId, editingCell.field, finalValue);
    setEditingCell(null);
  }, [editingCell, editValue, columns, onCellEdit]);

  const cancelEdit = useCallback(() => {
    setEditingCell(null);
    setEditValue("");
  }, []);

  const handleEditKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement | HTMLSelectElement>) => {
      if (e.key === "Enter") commitEdit();
      else if (e.key === "Escape") cancelEdit();
    },
    [commitEdit, cancelEdit],
  );

  /* Sort handler */
  const handleSort = useCallback(
    (colKey: string) => {
      if (!onSort) return;
      if (sortBy === colKey) {
        onSort(colKey, sortOrder === "asc" ? "desc" : "asc");
      } else {
        onSort(colKey, "asc");
      }
    },
    [sortBy, sortOrder, onSort],
  );

  /* Get display value for a cell (check editedCells overlay first) */
  const getCellValue = useCallback(
    (row: Record<string, unknown>, field: string) => {
      const rowId = String(row[rowIdKey]);
      const edited = editedCells?.get(rowId);
      if (edited && field in edited) return edited[field];
      return row[field];
    },
    [rowIdKey, editedCells],
  );

  const isRowEdited = useCallback(
    (rowId: string) => editedCells?.has(rowId) ?? false,
    [editedCells],
  );

  const isRowDeleted = useCallback(
    (rowId: string) => deletedIds?.has(rowId) ?? false,
    [deletedIds],
  );

  /* ---------- Render ---------- */

  if (data.length === 0) {
    return (
      <div className="data-table-empty">
        {emptyIcon}
        <p>{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className="data-table-container">
      {/* Scrollable area */}
      <div ref={scrollRef} style={{ maxHeight, overflow: "auto" }}>
        {/* Header */}
        <div
          className="data-table-header"
          style={{
            position: "sticky",
            top: 0,
            zIndex: 10,
            display: "flex",
            minHeight: 40,
          }}
        >
          {selectable && (
            <div className="data-table-cell data-table-cell-checkbox" style={{ width: 44, flexShrink: 0 }}>
              <input
                type="checkbox"
                checked={allSelected}
                onChange={handleSelectAll}
                aria-label="Select all rows"
              />
            </div>
          )}
          {columns.map((col) => (
            <div
              key={col.key}
              className={`data-table-cell data-table-header-cell${col.sortable ? " sortable" : ""}${col.sticky ? " sticky-col" : ""}`}
              style={{
                width: col.width || 140,
                minWidth: col.minWidth || 80,
                flexShrink: 0,
              }}
              onClick={() => col.sortable && handleSort(col.key)}
              role={col.sortable ? "button" : undefined}
              tabIndex={col.sortable ? 0 : undefined}
            >
              <span className="header-label">{col.label}</span>
              {col.sortable && sortBy === col.key && (
                <span className="sort-icon">
                  {sortOrder === "asc" ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                </span>
              )}
            </div>
          ))}
          {(editable || onDeleteRow) && (
            <div className="data-table-cell data-table-header-cell" style={{ width: 80, flexShrink: 0 }}>
              Actions
            </div>
          )}
        </div>

        {/* Rows — regular DOM, no virtualizer */}
        {displayData.map((row) => {
          const rowId = String(row[rowIdKey]);
          const selected = selectedIds?.has(rowId) ?? false;
          const deleted = isRowDeleted(rowId);
          const edited = isRowEdited(rowId);
          const highlight = highlightedRows?.get(rowId);

          let rowClass = "data-table-row";
          if (selected) rowClass += " selected";
          if (deleted) rowClass += " deleted";
          if (edited) rowClass += " edited";
          if (highlight) rowClass += ` highlight-${highlight}`;

          return (
            <div
              key={rowId}
              className={rowClass}
              style={{ display: "flex" }}
            >
              {selectable && (
                <div className="data-table-cell data-table-cell-checkbox" style={{ width: 44, flexShrink: 0 }}>
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={(e) => handleRowSelect(rowId, e.nativeEvent instanceof MouseEvent && (e.nativeEvent as MouseEvent).shiftKey)}
                    aria-label={`Select row ${rowId}`}
                  />
                </div>
              )}
              {columns.map((col) => {
                const value = getCellValue(row, col.key);
                const isEditing =
                  editingCell?.rowId === rowId && editingCell?.field === col.key;
                const cellEdited = editedCells?.get(rowId)?.[col.key] !== undefined;

                let cellClass = "data-table-cell";
                if (col.type === "float" || col.type === "int") cellClass += " numeric";
                if (col.sticky) cellClass += " sticky-col";
                if (isEditing) cellClass += " editing";
                if (cellEdited) cellClass += " cell-edited";
                if (editingRowId === rowId && col.editable) cellClass += " edit-ready";

                return (
                  <div
                    key={col.key}
                    className={cellClass}
                    style={{
                      width: col.width || 140,
                      minWidth: col.minWidth || 80,
                      flexShrink: 0,
                    }}
                    onDoubleClick={() => {
                      if (col.editable && editable && !deleted) {
                        startEditing(rowId, col.key, value);
                      }
                    }}
                    onClick={() => {
                      // Single click enters edit mode for editable cells
                      if (col.editable && editable && !deleted) {
                        startEditing(rowId, col.key, value);
                      }
                    }}
                  >
                    {isEditing ? (
                      col.options ? (
                        <select
                          className="cell-edit-input"
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onBlur={commitEdit}
                          onKeyDown={handleEditKeyDown}
                          autoFocus
                        >
                          <option value="">—</option>
                          {col.options.map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          className="cell-edit-input"
                          type={col.type === "float" || col.type === "int" ? "number" : "text"}
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onBlur={commitEdit}
                          onKeyDown={handleEditKeyDown}
                          autoFocus
                          step={col.type === "float" ? "any" : undefined}
                        />
                      )
                    ) : col.render ? (
                      col.render(value, row)
                    ) : col.type === "badge" ? (
                      <span className={`issue-badge ${String(value)}`}>
                        {String(value ?? "")}
                      </span>
                    ) : (
                      <span className="cell-text" title={String(value ?? "")}>
                        {value == null ? "—" : String(value)}
                      </span>
                    )}
                  </div>
                );
              })}
              {(editable || onDeleteRow) && (
                <div className="data-table-cell data-table-cell-actions" style={{ width: 80, flexShrink: 0 }}>
                  {editable && !deleted && (
                    <button
                      className={`action-btn${editingRowId === rowId ? " active" : ""}`}
                      onClick={() => {
                        setEditingRowId(editingRowId === rowId ? null : rowId);
                      }}
                      title={editingRowId === rowId ? "Exit edit mode" : "Edit row (double-click any cell)"}
                      aria-label="Toggle row edit mode"
                    >
                      <Pencil size={14} />
                    </button>
                  )}
                  {onDeleteRow && !deleted && (
                    <button
                      className="action-btn action-btn-danger"
                      onClick={() => onDeleteRow(rowId)}
                      title="Delete row"
                      aria-label="Delete row"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Pagination bar */}
      {paginate && totalPages >= 1 && (
        <div className="data-table-pagination">
          <div className="pagination-left">
            <span className="pagination-info">
              Showing {pageStart}-{pageEnd} of {totalRows} rows
              {totalPages > 1 && ` • Page ${activePage} of ${totalPages}`}
            </span>
          </div>
          <div className="pagination-center">
            {totalPages > 1 && (
              <>
                <button
                  className="btn-ghost btn-sm"
                  disabled={activePage <= 1}
                  onClick={() => goToPage(activePage - 1)}
                >
                  ← Prev
                </button>
                {/* Page numbers */}
                {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                  let page: number;
                  if (totalPages <= 7) {
                    page = i + 1;
                  } else if (activePage <= 4) {
                    page = i + 1;
                  } else if (activePage >= totalPages - 3) {
                    page = totalPages - 6 + i;
                  } else {
                    page = activePage - 3 + i;
                  }
                  return (
                    <button
                      key={page}
                      className={`btn-ghost btn-sm${page === activePage ? " btn-page-active" : ""}`}
                      onClick={() => goToPage(page)}
                    >
                      {page}
                    </button>
                  );
                })}
                <button
                  className="btn-ghost btn-sm"
                  disabled={activePage >= totalPages}
                  onClick={() => goToPage(activePage + 1)}
                >
                  Next →
                </button>
              </>
            )}
          </div>
          <div className="pagination-right">
            <label className="page-size-label">
              Show
              <select
                className="page-size-select"
                value={useExternal ? (externalPageSize ?? pageSize) : pageSize}
                onChange={(e) => handlePageSizeChange(Number(e.target.value))}
              >
                {pageSizeOptions.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              / page
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
