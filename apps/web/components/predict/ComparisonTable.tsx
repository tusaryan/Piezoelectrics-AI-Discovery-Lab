"use client";

/**
 * ComparisonTable — Side-by-side multi-material comparison.
 */

import { GitCompareArrows, X, Download, Trash2 } from "lucide-react";
import { usePredictStore } from "@/lib/store/predictStore";

export default function ComparisonTable() {
  const { comparisonList, removeFromComparison, clearComparison } =
    usePredictStore();

  if (comparisonList.length === 0) {
    return (
      <div className="predict-card">
        <div className="predict-card-title">
          <GitCompareArrows size={16} /> Multi-Material Comparison
        </div>
        <div className="predict-empty-state">
          <div className="predict-empty-icon">
            <GitCompareArrows size={24} />
          </div>
          <h3>No Materials to Compare</h3>
          <p>
            Run predictions and click &quot;Add to Comparison&quot; to build a
            side-by-side view of material properties.
          </p>
        </div>
      </div>
    );
  }

  const handleExportCSV = () => {
    const headers = [
      "Formula",
      "Type",
      "d33 (pC/N)",
      "d33 CI Lower",
      "d33 CI Upper",
      "Tc (°C)",
      "Tc CI Lower",
      "Tc CI Upper",
      "Hardness (HV)",
      "Use Case",
    ];

    const rows = comparisonList.map((entry) => {
      const p = entry.prediction;
      return [
        p.formula,
        p.is_composite ? "Composite" : "Bulk",
        p.d33?.value ?? "",
        p.d33?.ci_lower ?? "",
        p.d33?.ci_upper ?? "",
        p.tc?.value ?? "",
        p.tc?.ci_lower ?? "",
        p.tc?.ci_upper ?? "",
        p.hardness?.value ?? "",
        p.use_case?.name ?? "",
      ].join(",");
    });

    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `piezo_ai_comparison_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <GitCompareArrows size={16} /> Multi-Material Comparison (
        {comparisonList.length})
      </div>

      <div className="comparison-container">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Formula</th>
              <th>Type</th>
              <th>d₃₃ (pC/N)</th>
              <th>Tc (°C)</th>
              <th>Hardness (HV)</th>
              <th>Use Case</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {comparisonList.map((entry) => {
              const p = entry.prediction;
              return (
                <tr key={entry.id}>
                  <td className="formula-cell">{p.formula}</td>
                  <td>{p.is_composite ? "Composite" : "Bulk"}</td>
                  <td>
                    {p.d33?.value != null ? (
                      <>
                        {p.d33.value.toFixed(1)}
                        {p.d33.ci_lower != null && (
                          <span
                            style={{
                              fontSize: 10,
                              color: "var(--text-muted)",
                              marginLeft: 4,
                            }}
                          >
                            ±{((p.d33.ci_upper! - p.d33.ci_lower!) / 2).toFixed(1)}
                          </span>
                        )}
                      </>
                    ) : (
                      "–"
                    )}
                  </td>
                  <td>
                    {p.tc?.value != null ? p.tc.value.toFixed(1) : "–"}
                  </td>
                  <td>
                    {p.hardness?.value != null
                      ? p.hardness.value.toFixed(1)
                      : "–"}
                  </td>
                  <td>{p.use_case?.name ?? "–"}</td>
                  <td>
                    <button
                      className="comparison-remove-btn"
                      onClick={() => removeFromComparison(entry.id)}
                      title="Remove"
                    >
                      <X size={12} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="comparison-actions">
        <button
          className="comparison-action-btn primary"
          onClick={handleExportCSV}
        >
          <Download size={14} /> Export CSV
        </button>
        <button className="comparison-action-btn" onClick={clearComparison}>
          <Trash2 size={14} /> Clear All
        </button>
      </div>
    </div>
  );
}
