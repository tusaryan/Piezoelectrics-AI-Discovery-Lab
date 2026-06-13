"use client";

/**
 * Piezo.AI — Dataset Comparison View
 * =====================================
 * Tabs: Source Data | Parsed Preview | Comparison (side-by-side)
 *
 * Source Data: Shows the raw uploaded dataset fields (formula, d33, tc, etc.)
 * Parsed Preview: Shows elemental decomposition parsed ON-DEMAND from DB materials
 *   via POST /dashboard/datasets/{id}/parse — no training needed.
 * Comparison: Side-by-side source vs parsed for manual verification.
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
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import { getMaterials, type MaterialRow } from "@/lib/api/datasets";
import { parseDataset } from "@/lib/api/dashboard";

/* ---------- All displayable fields (source) ---------- */

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

/* ---------- Helper: format numeric cell ---------- */
function formatCell(val: unknown): string {
  if (val == null || val === "") return "—";
  if (typeof val === "number") {
    if (Number.isInteger(val)) return String(val);
    return val.toFixed(4);
  }
  return String(val);
}

/* ---------- Component ---------- */

export default function DatasetComparisonView() {
  const {
    activeDatasetId,
    comparisonTab,
    setComparisonTab,
  } = useDatasetStore();

  // Source data (raw uploaded dataset)
  const [materials, setMaterials] = useState<MaterialRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Parsed data — fetched on-demand via POST /dashboard/datasets/{id}/parse
  const [parsedRows, setParsedRows] = useState<Record<string, unknown>[]>([]);
  const [parsedColumns, setParsedColumns] = useState<string[]>([]);
  const [parsedFound, setParsedFound] = useState(false);
  const [parsedLoading, setParsedLoading] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [parseStats, setParseStats] = useState<{ total: number; skipped: number } | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [searchInput, setSearchInput] = useState("");

  const [visibleFields, setVisibleFields] = useState<Set<string>>(new Set(ALL_FIELDS.map(f => f.key)));
  const [searchFields, setSearchFields] = useState<Set<string>>(new Set(ALL_FIELDS.map(f => f.key)));
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Parsed columns visibility
  const [visibleParsedCols, setVisibleParsedCols] = useState<Set<string>>(new Set());

  /* Load source materials */
  useEffect(() => {
    if (!activeDatasetId) {
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setLoadError(null);
    getMaterials(activeDatasetId, { page: 1, page_size: 5000 })
      .then((r) => {
        setMaterials(r.items);
      })
      .catch((err) => {
        console.error("[ComparisonView] Failed to load materials:", err);
        setLoadError(String(err));
        setMaterials([]);
      })
      .finally(() => setIsLoading(false));
  }, [activeDatasetId]);

  /* Auto-parse on mount — parse formulas on-demand from DB */
  const runParse = useCallback(async () => {
    if (!activeDatasetId) return;
    setParsedLoading(true);
    setParseError(null);
    try {
      const result = await parseDataset(activeDatasetId, true);
      setParsedRows(result.rows);
      setParsedColumns(result.columns);
      setParsedFound(result.rows.length > 0);
      setVisibleParsedCols(new Set(result.columns));
      setParseStats({ total: result.total_parsed, skipped: result.total_skipped });
    } catch (err) {
      console.error("[ComparisonView] Parse failed:", err);
      setParseError(err instanceof Error ? err.message : String(err));
      setParsedFound(false);
      setParsedRows([]);
      setParsedColumns([]);
    } finally {
      setParsedLoading(false);
    }
  }, [activeDatasetId]);

  useEffect(() => {
    runParse();
  }, [runParse]);

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

  /* Visible columns */
  const displayFields = useMemo(
    () => ALL_FIELDS.filter((f) => fieldsWithData.has(f.key) && visibleFields.has(f.key)),
    [fieldsWithData, visibleFields],
  );
  const sourceFields = useMemo(
    () => displayFields.filter((f) => f.category !== "validation"),
    [displayFields],
  );

  /* Filter materials by search */
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

  /* Paginate */
  const paginatedData = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filteredMaterials.slice(start, start + pageSize);
  }, [filteredMaterials, page, pageSize]);

  /* Build uid→parsedRow map */
  const parsedByUid = useMemo(() => {
    const map = new Map<string, Record<string, unknown>>();
    for (const row of parsedRows) {
      const uid = String(row.uid);
      if (uid) map.set(uid, row);
    }
    return map;
  }, [parsedRows]);

  /* Paginate parsed rows */
  const paginatedParsed = useMemo(() => {
    const start = (page - 1) * pageSize;
    return parsedRows.slice(start, start + pageSize);
  }, [parsedRows, page, pageSize]);

  /* Visible parsed columns */
  const displayParsedCols = useMemo(
    () => parsedColumns.filter((c) => visibleParsedCols.has(c)),
    [parsedColumns, visibleParsedCols],
  );

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
    setVisibleParsedCols(new Set(parsedColumns));
    setPage(1);
  }, [parsedColumns]);

  /* Cell value renderer for source data */
  const renderSourceVal = (m: MaterialRow, key: keyof MaterialRow) => {
    const v = m[key];
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

      {/* Search + Filter bar (for source tabs) */}
      {(comparisonTab === "source" || comparisonTab === "comparison") && (
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

          <MultiSelectFilter
            label="Search In"
            icon={<Search size={12} />}
            options={ALL_FIELDS.map(f => ({ key: f.key, label: f.label }))}
            selected={searchFields}
            onChange={(next) => { setSearchFields(next); setPage(1); }}
          />

          <MultiSelectFilter
            label="Columns"
            icon={<Filter size={12} />}
            options={ALL_FIELDS.filter(f => fieldsWithData.has(f.key)).map(f => ({ key: f.key, label: f.label }))}
            selected={visibleFields}
            onChange={setVisibleFields}
            required={new Set(["uid", "formula"])}
          />

          <button className="btn-ghost btn-sm" onClick={handleResetFilters} title="Reset all filters">
            <RotateCcw size={14} /> Reset
          </button>
        </div>
      )}

      {/* Parsed column filter (for parsed tab) */}
      {comparisonTab === "parsed" && parsedFound && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <MultiSelectFilter
            label="Columns"
            icon={<Filter size={12} />}
            options={parsedColumns.map(c => ({ key: c, label: c }))}
            selected={visibleParsedCols}
            onChange={setVisibleParsedCols}
            required={new Set(["uid", "formula"])}
          />
          <button className="btn-ghost btn-sm" onClick={() => setVisibleParsedCols(new Set(parsedColumns))} title="Show all columns">
            <RotateCcw size={14} /> Show All
          </button>
          <button className="btn-ghost btn-sm" onClick={runParse} disabled={parsedLoading} title="Re-parse formulas">
            <RefreshCw size={14} className={parsedLoading ? "spin" : ""} /> Re-parse
          </button>
        </div>
      )}

      {/* Source Data tab */}
      {comparisonTab === "source" && (
        <>
          <div className="comparison-info">
            <Info size={14} />
            <span>Source data as uploaded. Shows the raw dataset fields before elemental decomposition.</span>
          </div>
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
                        {renderSourceVal(m, f.key)}
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

      {/* Parsed Preview tab — on-demand parsing, NO training required */}
      {comparisonTab === "parsed" && (
        <>
          {parsedLoading ? (
            <div className="review-loading" style={{ padding: 32 }}>
              <Loader2 size={20} className="spin" />
              <p>Parsing formulas (stoichiometry + elemental decomposition)...</p>
            </div>
          ) : parseError ? (
            <div className="comparison-info" style={{ flexDirection: "column", gap: 8, padding: 24 }}>
              <AlertTriangle size={20} style={{ color: "var(--error)" }} />
              <span>
                <strong>Parsing failed.</strong> {parseError}
              </span>
              <button className="btn-ghost btn-sm" onClick={runParse} style={{ marginTop: 8 }}>
                <RefreshCw size={14} /> Retry
              </button>
            </div>
          ) : !parsedFound ? (
            <div className="comparison-info" style={{ flexDirection: "column", gap: 8, padding: 24 }}>
              <AlertTriangle size={20} style={{ color: "var(--warning)" }} />
              <span>
                <strong>No parseable formulas found.</strong> All formulas may have unsupported elements or invalid syntax.
              </span>
              <button className="btn-ghost btn-sm" onClick={runParse} style={{ marginTop: 8 }}>
                <RefreshCw size={14} /> Retry
              </button>
            </div>
          ) : (
            <>
              <div className="comparison-info">
                <Info size={14} />
                <span>
                  Parsed elemental decomposition (on-demand). Shows element fractions (e.g., Na_frac, K_frac),
                  stoichiometry, and carried-over material properties.
                  {parseStats && (
                    <> — <strong>{parseStats.total}</strong> parsed, <strong>{parseStats.skipped}</strong> skipped.</>
                  )}
                </span>
              </div>
              <div className="comparison-table-wrapper">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      {displayParsedCols.map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedParsed.map((row, idx) => (
                      <tr key={row.uid as string || idx}>
                        {displayParsedCols.map((col) => (
                          <td key={col} className={typeof row[col] === "number" ? "numeric" : ""}>
                            {formatCell(row[col])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <TablePagination total={parsedRows.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={handlePageSizeChange} />
            </>
          )}
        </>
      )}

      {/* Comparison tab — side-by-side source vs parsed */}
      {comparisonTab === "comparison" && (
        <>
          {!parsedFound && (
            <div className="comparison-info" style={{ background: "color-mix(in srgb, #F59E0B 8%, transparent)" }}>
              <AlertTriangle size={14} style={{ color: "#F59E0B" }} />
              <span>
                Parsed data not available yet. Switch to &quot;Parsed Preview&quot; tab to trigger on-demand parsing.
              </span>
            </div>
          )}
          <div className="comparison-split">
            <div className="comparison-pane">
              <div className="comparison-pane-header">Source (Uploaded)</div>
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
                          <td key={f.key}>{renderSourceVal(m, f.key)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="comparison-pane">
              <div className="comparison-pane-header">Parsed (Elemental)</div>
              <div className="comparison-table-wrapper">
                {parsedFound ? (
                  <table className="comparison-table compact">
                    <thead>
                      <tr>
                        {displayParsedCols.slice(0, 10).map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedData.map((m) => {
                        const parsed = parsedByUid.get(String(m.uid));
                        return (
                          <tr key={m.id} className={!parsed ? "highlight-warning" : ""}>
                            {displayParsedCols.slice(0, 10).map((col) => (
                              <td key={col} className={typeof parsed?.[col] === "number" ? "numeric" : ""}>
                                {parsed ? formatCell(parsed[col]) : "—"}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
                    Switch to &quot;Parsed Preview&quot; to parse formulas first
                  </div>
                )}
              </div>
            </div>
          </div>
          <TablePagination total={filteredMaterials.length} page={page} pageSize={pageSize} onPageChange={setPage} onPageSizeChange={handlePageSizeChange} />
        </>
      )}
    </div>
  );
}
