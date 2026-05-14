"use client";

import { useState, useMemo } from "react";
import { Trophy, Download, ChevronDown, ChevronUp } from "lucide-react";
import { useOptimizationStore } from "@/lib/store/optimizationStore";

const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃",
  tc: "Tc",
  vickers_hardness: "HV",
};

const TARGET_UNITS: Record<string, string> = {
  d33: "pC/N",
  tc: "°C",
  vickers_hardness: "HV",
};

export default function SolutionTable() {
  const { solutions, optimizationStats } = useOptimizationStore();
  const [sortField, setSortField] = useState("rank");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [showAll, setShowAll] = useState(false);

  const targets = optimizationStats?.targets_optimized || [];

  const sorted = useMemo(() => {
    const items = [...solutions];
    items.sort((a, b) => {
      let va: number, vb: number;
      if (sortField === "rank") {
        va = a.rank;
        vb = b.rank;
      } else if (sortField === "formula") {
        return sortDir === "asc"
          ? a.formula_approx.localeCompare(b.formula_approx)
          : b.formula_approx.localeCompare(a.formula_approx);
      } else if (sortField === "use_case") {
        return sortDir === "asc"
          ? a.use_case_tag.localeCompare(b.use_case_tag)
          : b.use_case_tag.localeCompare(a.use_case_tag);
      } else {
        va = a.predicted[sortField] ?? 0;
        vb = b.predicted[sortField] ?? 0;
      }
      return sortDir === "asc" ? va - vb : vb - va;
    });
    return items;
  }, [solutions, sortField, sortDir]);

  const displayed = showAll ? sorted : sorted.slice(0, 20);

  if (solutions.length === 0) return null;

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir(field === "rank" ? "asc" : "desc");
    }
  };

  const SortIcon = ({ field }: { field: string }) => {
    if (sortField !== field) return null;
    return sortDir === "asc" ? (
      <ChevronUp size={12} />
    ) : (
      <ChevronDown size={12} />
    );
  };

  const handleExportCSV = () => {
    const headers = [
      "rank",
      "formula",
      ...targets.map((t) => `${t}_predicted`),
      "use_case",
      ...solutions[0]
        ? Object.keys(solutions[0].composition).map((e) => `frac_${e}`)
        : [],
    ];
    const rows = sorted.map((s) => [
      s.rank,
      s.formula_approx,
      ...targets.map((t) => (s.predicted[t] ?? "").toString()),
      s.use_case_tag,
      ...Object.values(s.composition).map((v) => v.toFixed(4)),
    ]);
    const csv =
      headers.join(",") + "\n" + rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pareto_solutions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="opt-card">
      <div className="opt-card-header">
        <Trophy size={18} />
        <h3>Pareto-Optimal Solutions</h3>
        <span className="opt-badge">{solutions.length} solutions</span>
        <button className="opt-export-btn" onClick={handleExportCSV}>
          <Download size={14} />
          Export CSV
        </button>
      </div>

      <div className="opt-table-wrapper">
        <table className="opt-table">
          <thead>
            <tr>
              <th onClick={() => handleSort("rank")} className="opt-th-sortable">
                # <SortIcon field="rank" />
              </th>
              <th
                onClick={() => handleSort("formula")}
                className="opt-th-sortable"
              >
                Formula <SortIcon field="formula" />
              </th>
              {targets.map((t) => (
                <th
                  key={t}
                  onClick={() => handleSort(t)}
                  className="opt-th-sortable"
                >
                  {TARGET_LABELS[t] || t}{" "}
                  <span className="opt-th-unit">({TARGET_UNITS[t] || ""})</span>
                  <SortIcon field={t} />
                </th>
              ))}
              <th
                onClick={() => handleSort("use_case")}
                className="opt-th-sortable"
              >
                Use Case <SortIcon field="use_case" />
              </th>
            </tr>
          </thead>
          <tbody>
            {displayed.map((s) => (
              <tr key={s.rank} className="opt-tr">
                <td className="opt-rank-cell">
                  {s.rank <= 3 ? (
                    <span className={`opt-rank-badge rank-${s.rank}`}>
                      {s.rank}
                    </span>
                  ) : (
                    s.rank
                  )}
                </td>
                <td className="opt-formula-cell">{s.formula_approx}</td>
                {targets.map((t) => (
                  <td key={t} className="opt-value-cell">
                    {(s.predicted[t] ?? 0).toFixed(1)}
                  </td>
                ))}
                <td>
                  <span
                    className="opt-usecase-tag"
                    style={{
                      background: `${s.use_case_color}20`,
                      color: s.use_case_color,
                      borderColor: `${s.use_case_color}40`,
                    }}
                  >
                    {s.use_case_tag}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {solutions.length > 20 && (
        <button
          className="opt-show-more-btn"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll
            ? `Show less (top 20)`
            : `Show all ${solutions.length} solutions`}
        </button>
      )}
    </div>
  );
}
