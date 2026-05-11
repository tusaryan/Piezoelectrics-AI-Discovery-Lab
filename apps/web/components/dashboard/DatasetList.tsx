"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, useRef } from "react";
import {
  Database,
  Eye,
  Download,
  CheckCircle,
  Clock,
  Rows3,
  Columns3,
  MoreVertical,
  Pencil,
  Copy,
  Trash2,
  Check,
  X,
  FileSearch,
  Loader2,
} from "lucide-react";
import { listDatasets, type DatasetSummary } from "@/lib/api/datasets";
import {
  renameDashboardDataset,
  copyDashboardDataset,
  deleteDashboardDataset,
  parseDataset,
} from "@/lib/api/dashboard";
import { useDatasetStore } from "@/lib/store/datasetStore";
import { useUIStore } from "@/lib/store/uiStore";
import { APP_CONFIG } from "@/lib/constants";

function formatExactTime(isoStr: string): string {
  const d = new Date(isoStr);
  return d.toLocaleDateString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
  }) + ", " + d.toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

export default function DatasetList() {
  const router = useRouter();
  const { setActiveDatasetId } = useDatasetStore();
  const strictMode = useUIStore((s: { strictFormulaMode: boolean }) => s.strictFormulaMode);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [parsingId, setParsingId] = useState<string | null>(null);
  const [parseResult, setParseResult] = useState<{
    total_parsed: number;
    total_skipped: number;
  } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  // Close menu on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
      }
    }
    if (menuOpenId) {
      document.addEventListener("mousedown", handleClick);
      return () => document.removeEventListener("mousedown", handleClick);
    }
  }, [menuOpenId]);

  async function loadDatasets() {
    try {
      const data = await listDatasets();
      setDatasets(data);
    } catch {
      /* silent on dashboard */
    } finally {
      setLoading(false);
    }
  }

  function handleView(id: string) {
    setActiveDatasetId(id);
    router.push("/dataset");
  }

  function handleDownload(id: string, name: string) {
    const url = `${APP_CONFIG.api.baseUrl}/api/v1/datasets/${id}/materials?page=1&page_size=10000`;
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        const items = data.items || data;
        if (!Array.isArray(items) || items.length === 0) return;
        const headers = Object.keys(items[0]);
        const csvRows = [
          headers.join(","),
          ...items.map((row: Record<string, unknown>) =>
            headers.map((h) => {
              const val = row[h];
              if (val === null || val === undefined) return "";
              const str = String(val);
              return str.includes(",") ? `"${str}"` : str;
            }).join(",")
          ),
        ];
        const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${name.replace(/\.csv$/i, "")}.csv`;
        a.click();
        URL.revokeObjectURL(a.href);
      })
      .catch(() => { /* silent */ });
  }

  // Rename
  function startRename(ds: DatasetSummary) {
    setEditingId(ds.id);
    setEditName(ds.display_name);
    setMenuOpenId(null);
  }

  async function submitRename(id: string) {
    if (editName.trim()) {
      try {
        await renameDashboardDataset(id, editName.trim());
        setDatasets((prev) =>
          prev.map((ds) => ds.id === id ? { ...ds, display_name: editName.trim() } : ds)
        );
      } catch { /* silent */ }
    }
    setEditingId(null);
  }

  // Copy
  async function handleCopy(id: string) {
    setMenuOpenId(null);
    try {
      await copyDashboardDataset(id);
      await loadDatasets(); // Refresh list
    } catch { /* silent */ }
  }

  // Delete
  async function handleDelete(id: string) {
    try {
      await deleteDashboardDataset(id);
      setDatasets((prev) => prev.filter((ds) => ds.id !== id));
    } catch { /* silent */ }
    setConfirmDeleteId(null);
  }

  // Parse on demand
  async function handleParse(id: string) {
    setParsingId(id);
    setMenuOpenId(null);
    try {
      const result = await parseDataset(id, strictMode);
      setParseResult({
        total_parsed: result.total_parsed,
        total_skipped: result.total_skipped,
      });
      setTimeout(() => setParseResult(null), 4000);
    } catch { /* silent */ }
    setParsingId(null);
  }

  if (loading) {
    return (
      <div className="dashboard-section">
        <h2 className="section-title">
          <Database size={18} /> Datasets
        </h2>
        <div className="skeleton-list">
          {[1, 2, 3].map((i) => (
            <div key={i} className="skeleton-row" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-section">
      <div className="section-header">
        <h2 className="section-title">
          <Database size={18} /> Datasets
        </h2>
        <span className="section-count">{datasets.length}</span>
      </div>

      {/* Parse result notification */}
      {parseResult && (
        <div className="parse-result-notice">
          <CheckCircle size={14} />
          Parsed {parseResult.total_parsed} formulas
          {parseResult.total_skipped > 0 && ` (${parseResult.total_skipped} skipped)`}
        </div>
      )}

      {datasets.length === 0 ? (
        <div className="empty-state">
          <Database size={32} />
          <p>No datasets uploaded yet</p>
          <button
            className="empty-state-btn"
            onClick={() => router.push("/dataset")}
          >
            Upload Dataset
          </button>
        </div>
      ) : (
        <div className="dataset-table-wrapper">
          <table className="dashboard-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Rows</th>
                <th>Columns</th>
                <th>Uploaded</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((ds) => (
                <tr key={ds.id}>
                  <td className="td-name">
                    {editingId === ds.id ? (
                      <div className="inline-rename">
                        <input
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") submitRename(ds.id);
                            if (e.key === "Escape") setEditingId(null);
                          }}
                          autoFocus
                          className="inline-rename-input"
                        />
                        <button onClick={() => submitRename(ds.id)} className="rename-confirm-sm">
                          <Check size={12} />
                        </button>
                        <button onClick={() => setEditingId(null)} className="rename-cancel-sm">
                          <X size={12} />
                        </button>
                      </div>
                    ) : (
                      ds.display_name
                    )}
                  </td>
                  <td>
                    <span
                      className={`status-badge ${ds.status === "ready" ? "status-ready" : "status-pending"}`}
                    >
                      {ds.status === "ready" ? (
                        <CheckCircle size={12} />
                      ) : (
                        <Clock size={12} />
                      )}
                      {ds.status}
                    </span>
                  </td>
                  <td>
                    <span className="meta-chip">
                      <Rows3 size={12} /> {ds.total_rows}
                    </span>
                  </td>
                  <td>
                    <span className="meta-chip">
                      <Columns3 size={12} /> {ds.total_columns}
                    </span>
                  </td>
                  <td className="td-date">
                    {formatExactTime(ds.uploaded_at)}
                  </td>
                  <td className="td-actions">
                    <button
                      className="action-btn action-view"
                      onClick={() => handleView(ds.id)}
                      title="View in Explorer"
                    >
                      <Eye size={14} />
                    </button>
                    <button
                      className="action-btn action-download"
                      onClick={() => handleDownload(ds.id, ds.display_name)}
                      title="Download CSV"
                    >
                      <Download size={14} />
                    </button>

                    {/* Three-dot menu */}
                    <div
                      className="dataset-menu-wrapper"
                      ref={menuOpenId === ds.id ? menuRef : undefined}
                    >
                      <button
                        className="action-btn action-menu"
                        onClick={() => setMenuOpenId(menuOpenId === ds.id ? null : ds.id)}
                        title="More actions"
                      >
                        <MoreVertical size={14} />
                      </button>
                      {menuOpenId === ds.id && (
                        <div className="dataset-kebab-menu">
                          <button onClick={() => startRename(ds)}>
                            <Pencil size={13} /> Rename
                          </button>
                          <button onClick={() => handleCopy(ds.id)}>
                            <Copy size={13} /> Copy
                          </button>
                          <button onClick={() => handleParse(ds.id)} disabled={parsingId === ds.id}>
                            {parsingId === ds.id ? (
                              <><Loader2 size={13} className="spin" /> Parsing…</>
                            ) : (
                              <><FileSearch size={13} /> Parse Formulas</>
                            )}
                          </button>
                          {confirmDeleteId === ds.id ? (
                            <div className="kebab-delete-confirm">
                              <span>Delete?</span>
                              <button className="confirm-yes-sm" onClick={() => handleDelete(ds.id)}>Yes</button>
                              <button className="confirm-no-sm" onClick={() => setConfirmDeleteId(null)}>No</button>
                            </div>
                          ) : (
                            <button
                              className="kebab-delete"
                              onClick={() => { setConfirmDeleteId(ds.id); }}
                            >
                              <Trash2 size={13} /> Delete
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
