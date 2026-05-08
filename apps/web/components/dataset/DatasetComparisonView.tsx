"use client";

/**
 * Piezo.AI — Dataset Comparison View
 * =====================================
 * Tabs: Source Data | Parsed Preview | Comparison (side-by-side)
 *
 * S2: Shows all mapped fields + formula validation.
 * S3: Will add parsed elemental compositions & feature vectors.
 *
 * Features:
 * - Multi-select search field filter
 * - Column visibility toggles with select/deselect all
 * - Client-side pagination (25/50/100)
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  GitCompare,
  Search,
  Loader2,
  Info,
  Filter,
  X,
  RotateCcw,
  ChevronDown,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import { getMaterials, type MaterialRow } from "@/lib/api/datasets";

/* ---------- All displayable fields ---------- */

const ALL_FIELDS: { key: keyof MaterialRow; label: string; category: string }[] = [
  { key: "uid", label: "UID", category: "core" },
  { key: "formula", label: "Formula", category: "core" },
  { key: "d33", label: "d₃₃", category: "core" },
  { key: "tc", label: "Tc", category: "core" },
  { key: "vickers_hardness", label: "Hardness", category: "core" },
  { key: "qm", label: "Qm", category: "core" },
  { key: "kp", label: "kp", category: "core" },
  { key: "relative_density_pct", label: "Density%", category: "core" },
  { key: "sintering_temp_c", label: "Sint.Temp", category: "processing" },
  { key: "sintering_method", label: "Sint.Method", category: "processing" },
  { key: "ceramic_type", label: "Ceramic Type", category: "processing" },
  { key: "fabrication_method", label: "Fabrication", category: "processing" },
  { key: "matrix_type", label: "Matrix", category: "composite" },
  { key: "filler_wt_pct", label: "Filler%", category: "composite" },
  { key: "particle_morphology", label: "Morphology", category: "composite" },
  { key: "particle_size_nm", label: "Size(nm)", category: "composite" },
  { key: "surface_treatment", label: "Treatment", category: "composite" },
  { key: "parse_status", label: "Status", category: "validation" },
  { key: "parse_warnings", label: "Warnings", category: "validation" },
  { key: "source_doi", label: "DOI", category: "traceability" },
  { key: "source_notes", label: "Notes", category: "traceability" },
];


/* ---------- Status badge ---------- */

function StatusBadge({ status }: { status: string }) {
  const cls = status === "success" ? "status-success"
    : status === "error" ? "status-error"
    : status === "unsupported_elements" ? "status-warning"
    : "status-muted";
  return <span className={`comparison-status ${cls}`}>{status}</span>;
}

/* ---------- Multi-select filter dropdown ---------- */

function MultiSelectFilter({
  label,
  icon,
  options,
  selected,
  onChange,
  required,
}: {
  label: string;
  icon: ReactNode;
  options: { key: string; label: string }[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
  required?: Set<string>;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  /* Close on outside click */
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const allSelected = options.every(o => selected.has(o.key));
  const count = selected.size;

  return (
    <div className="filter-dropdown" ref={ref}>
      <button
        className={`btn-ghost btn-sm${open ? " active" : ""}`}
        onClick={() => setOpen(!open)}
        style={open ? { color: "var(--primary)", background: "var(--primary-glow)" } : {}}
      >
        {icon}
        {label}
        {count < options.length && <span style={{ fontSize: "0.625rem", opacity: 0.7, marginLeft: 2 }}>({count})</span>}
        <ChevronDown size={12} />
      </button>
      {open && (
        <div className="filter-dropdown-panel">
          <div className="filter-dropdown-actions">
            <button onClick={() => onChange(new Set(options.map(o => o.key)))}>Select All</button>
            <button onClick={() => onChange(new Set(required || []))}>Deselect All</button>
          </div>
          {options.map((opt) => {
            const isRequired = required?.has(opt.key);
            return (
              <label key={opt.key} className="filter-dropdown-item">
                <input
                  type="checkbox"
                  checked={selected.has(opt.key)}
                  disabled={isRequired}
                  onChange={() => {
                    const next = new Set(selected);
                    if (next.has(opt.key)) {
                      if (!isRequired) next.delete(opt.key);
                    } else {
                      next.add(opt.key);
                    }
                    onChange(next);
                  }}
                />
                {opt.label}
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ---------- Pagination ---------- */

function TablePagination({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: {
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (p: number) => void;
  onPageSizeChange: (s: number) => void;
}) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (total === 0) return null;
  return (
    <div className="data-table-pagination">
      <div className="pagination-left">
        <span className="pagination-info">
          {total} records{totalPages > 1 && ` • Page ${page} of ${totalPages}`}
        </span>
      </div>
      <div className="pagination-center">
        {totalPages > 1 && (
          <>
            <button className="btn-ghost btn-sm" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>← Prev</button>
            {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
              let p: number;
              if (totalPages <= 7) p = i + 1;
              else if (page <= 4) p = i + 1;
              else if (page >= totalPages - 3) p = totalPages - 6 + i;
              else p = page - 3 + i;
              return (
                <button key={p} className={`btn-ghost btn-sm${p === page ? " btn-page-active" : ""}`} onClick={() => onPageChange(p)}>
                  {p}
                </button>
              );
            })}
            <button className="btn-ghost btn-sm" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>Next →</button>
          </>
        )}
      </div>
      <div className="pagination-right">
        <label className="page-size-label">
          Show
          <select className="page-size-select" value={pageSize} onChange={(e) => onPageSizeChange(Number(e.target.value))}>
            {[25, 50, 100].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          / page
        </label>
      </div>
    </div>
  );
}

/* ---------- Component ---------- */

export default function DatasetComparisonView() {
  const {
    activeDatasetId,
    comparisonTab,
    setComparisonTab,
  } = useDatasetStore();

  const [materials, setMaterials] = useState<MaterialRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchInput, setSearchInput] = useState("");

  const [visibleFields, setVisibleFields] = useState<Set<string>>(new Set(ALL_FIELDS.map(f => f.key)));
  const [searchFields, setSearchFields] = useState<Set<string>>(new Set(ALL_FIELDS.map(f => f.key)));
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  /* Load materials */
  useEffect(() => {
    if (!activeDatasetId) {
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setLoadError(null);
    getMaterials(activeDatasetId, { page: 1, page_size: 5000 })
      .then((r) => {
        console.log(`[ComparisonView] Loaded ${r.items.length} materials`);
        setMaterials(r.items);
      })
      .catch((err) => {
        console.error("[ComparisonView] Failed to load materials:", err);
        setLoadError(String(err));
        setMaterials([]);
      })
      .finally(() => setIsLoading(false));
  }, [activeDatasetId]);

  /* Debounced search */
  useEffect(() => {
    const t = setTimeout(() => { setSearchQuery(searchInput); setPage(1); }, 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  /* Determine which fields have data */
  const fieldsWithData = useMemo(() => {
    const hasData = new Set<string>();
    materials.forEach((m) => {
      ALL_FIELDS.forEach((f) => {
        const v = m[f.key];
        if (v != null && v !== "" && v !== "none") hasData.add(f.key);
      });
    });
    hasData.add("uid");
    hasData.add("formula");
    hasData.add("parse_status");
    return hasData;
  }, [materials]);

  /* Visible columns — intersection of fieldsWithData and user-selected */
  const displayFields = useMemo(
    () => ALL_FIELDS.filter((f) => fieldsWithData.has(f.key) && visibleFields.has(f.key)),
    [fieldsWithData, visibleFields],
  );
  const sourceFields = useMemo(
    () => displayFields.filter((f) => f.category !== "validation"),
    [displayFields],
  );
  const parsedFields = useMemo(() => displayFields, [displayFields]);

  /* Filter materials by search across intersection of Search-In and visible columns */
  const filteredMaterials = useMemo(() => {
    if (!searchQuery) return materials;
    const q = searchQuery.toLowerCase();
    const searchableVisible = new Set(
      ALL_FIELDS
        .map((f) => f.key)
        .filter((key) => searchFields.has(key) && visibleFields.has(key)),
    );
    if (searchableVisible.size === 0) return [];
    return materials.filter((m) => {
      return ALL_FIELDS.some((f) => {
        if (!searchableVisible.has(f.key)) return false;
        const v = m[f.key];
        return v != null && String(v).toLowerCase().includes(q);
      });
    });
  }, [materials, searchQuery, searchFields, visibleFields]);

  /* Paginate filtered results */
  const paginatedData = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filteredMaterials.slice(start, start + pageSize);
  }, [filteredMaterials, page, pageSize]);

  const handlePageSizeChange = useCallback((size: number) => {
    setPageSize(size);
    setPage(1);
  }, []);

  /* Reset all filters */
  const handleResetFilters = useCallback(() => {
    setSearchInput("");
    setSearchQuery("");
    setSearchFields(new Set(ALL_FIELDS.map(f => f.key)));
    setVisibleFields(new Set(ALL_FIELDS.map(f => f.key)));
    setPage(1);
  }, []);



  const getSnapshotVal = (m: MaterialRow, key: keyof MaterialRow, which: "source" | "parsed") => {
    const snap = which === "source" ? (m as any).source_row : (m as any).parsed_row;
    if (snap && typeof snap === "object" && key in snap) return snap[key as any];
    return (m as any)[key as any];
  };

  /* Cell value renderer */
  const renderVal = (m: MaterialRow, key: keyof MaterialRow, which: "source" | "parsed") => {
    const v = getSnapshotVal(m, key, which);
    if (key === "parse_status") return <StatusBadge status={String(v || "pending")} />;
    if (key === "parse_warnings") return v ? String(v) : "—";
    if (v == null || v === "") return <span style={{ color: "var(--text-muted)" }}>—</span>;
    return String(v);
  };

  if (isLoading) {
    return (
      <div className="comparison-container">
        <div className="review-loading">
          <Loader2 size={24} className="spin" />
          <p>Loading comparison data...</p>
        </div>
      </div>
    );
  }

  if (materials.length === 0) {
    return (
      <div className="comparison-container">
        <div className="comparison-empty">
          <GitCompare size={32} />
          {loadError ? (
            <>
              <p style={{ color: "var(--error)" }}>Failed to load materials</p>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{loadError}</p>
            </>
          ) : (
            <p>No materials found. Upload and map a dataset first.</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="comparison-container">
      {/* Sub-tabs */}
      <div className="comparison-tabs">
        {(["source", "parsed", "comparison"] as const).map((tab) => (
          <button
            key={tab}
            className={`comparison-tab${comparisonTab === tab ? " active" : ""}`}
            onClick={() => { setComparisonTab(tab); setPage(1); }}
          >
            {tab === "source" ? "Source Data" : tab === "parsed" ? "Parsed Preview" : "Comparison"}
          </button>
        ))}
      </div>

      {/* Search + Filter bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <div className="comparison-search" style={{ flex: "0 1 260px" }}>
          <Search size={16} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
          <input
            type="text"
            placeholder="Search..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
          />
          {searchInput && (
            <button onClick={() => setSearchInput("")} style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: 2 }}>
              <X size={14} />
            </button>
          )}
        </div>

        {/* Multi-select: search in which fields */}
        <MultiSelectFilter
          label="Search In"
          icon={<Search size={12} />}
          options={ALL_FIELDS.map(f => ({ key: f.key, label: f.label }))}
          selected={searchFields}
          onChange={(next) => { setSearchFields(next); setPage(1); }}
        />

        {/* Column visibility filter */}
        <MultiSelectFilter
          label="Columns"
          icon={<Filter size={12} />}
          options={ALL_FIELDS.filter(f => fieldsWithData.has(f.key)).map(f => ({ key: f.key, label: f.label }))}
          selected={visibleFields}
          onChange={setVisibleFields}
          required={new Set(["uid", "formula"])}
        />

        {/* Reset */}
        <button className="btn-ghost btn-sm" onClick={handleResetFilters} title="Reset all filters">
          <RotateCcw size={14} /> Reset
        </button>
      </div>

      {/* Info banner */}
      <div className="comparison-info">
        <Info size={14} />
        <span>Full parsed compositions (element fractions, features) will be available after training. Currently showing formula validation results.</span>
      </div>

      {/* Source Data tab */}
      {comparisonTab === "source" && (
        <>
          <div className="comparison-table-wrapper">
            <table className="comparison-table">
              <thead>
                <tr>
                  {sourceFields.map((f) => (
                    <th key={f.key}>{f.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((m) => (
                  <tr key={m.id}>
                    {sourceFields.map((f) => (
                      <td key={f.key} className={typeof m[f.key] === "number" ? "numeric" : ""}>
                        {renderVal(m, f.key, "source")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <TablePagination total={filteredMaterials.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={handlePageSizeChange} />
        </>
      )}

      {/* Parsed Preview tab */}
      {comparisonTab === "parsed" && (
        <>
          <div className="comparison-table-wrapper">
            <table className="comparison-table">
              <thead>
                <tr>
                  {parsedFields.map((f) => (
                    <th key={f.key}>{f.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((m) => (
                  <tr
                    key={m.id}
                    className={
                      m.parse_status === "error" ? "highlight-error"
                      : m.parse_status === "unsupported_elements" ? "highlight-warning"
                      : ""
                    }
                  >
                    {parsedFields.map((f) => (
                      <td key={f.key} className={typeof m[f.key] === "number" ? "numeric" : ""}>
                        {renderVal(m, f.key, "parsed")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <TablePagination total={filteredMaterials.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={handlePageSizeChange} />
        </>
      )}

      {/* Comparison tab — side-by-side */}
      {comparisonTab === "comparison" && (
        <>
          <div className="comparison-split">
            <div className="comparison-pane">
              <div className="comparison-pane-header">Source</div>
              <div className="comparison-table-wrapper">
                <table className="comparison-table compact">
                  <thead>
                    <tr>
                      {sourceFields.map((f) => (
                        <th key={f.key}>{f.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedData.map((m) => (
                      <tr key={m.id}>
                        {sourceFields.map((f) => (
                          <td key={f.key}>{renderVal(m, f.key, "source")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="comparison-pane">
              <div className="comparison-pane-header">Parsed</div>
              <div className="comparison-table-wrapper">
                <table className="comparison-table compact">
                  <thead>
                    <tr>
                      {parsedFields.map((f) => (
                        <th key={f.key}>{f.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedData.map((m) => (
                      <tr
                        key={m.id}
                        className={
                          m.parse_status === "error" ? "highlight-error"
                          : m.parse_status === "unsupported_elements" ? "highlight-warning"
                          : ""
                        }
                      >
                        {parsedFields.map((f) => (
                          <td key={f.key}>{renderVal(m, f.key, "parsed")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <TablePagination total={filteredMaterials.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={handlePageSizeChange} />
        </>
      )}
    </div>
  );
}
