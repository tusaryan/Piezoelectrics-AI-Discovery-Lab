"use client";

import { useState } from "react";
import { Atom, Plus, X, Search, AlertTriangle, CheckCircle } from "lucide-react";
import { useOptimizationStore } from "@/lib/store/optimizationStore";
import type { StructuralDescriptor } from "@/lib/api/optimization";
import FormulaValidationInput from "@/components/ui/FormulaValidationInput";

const STABILITY_COLORS: Record<string, string> = {
  "highly stable": "#10B981",
  "stable (ferroelectric)": "#6366F1",
  "stable (tilted octahedra)": "#3B82F6",
  "marginally stable": "#F59E0B",
  "unstable perovskite": "#EF4444",
  "face-sharing instability": "#EF4444",
  undetermined: "#6B7280",
};

function DescriptorCard({ desc }: { desc: StructuralDescriptor }) {
  // ... same as before
  if (!desc.is_valid) {
    return (
      <div className="opt-struct-card opt-struct-error">
        <div className="opt-struct-formula">{desc.formula}</div>
        <div className="opt-struct-error-msg">
          <AlertTriangle size={14} />
          {desc.error}
        </div>
      </div>
    );
  }

  return (
    <div className="opt-struct-card">
      <div className="opt-struct-header">
        <span className="opt-struct-formula">{desc.normalized_formula}</span>
        <span
          className="opt-struct-stability"
          style={{
            color: STABILITY_COLORS[desc.stability_class] || "#6B7280",
            borderColor: STABILITY_COLORS[desc.stability_class] || "#6B7280",
          }}
        >
          {desc.crystal_system} • {desc.stability_class}
        </span>
      </div>
      <div className="opt-struct-perovskite">
        {desc.is_perovskite_likely ? (
          <CheckCircle size={14} className="opt-perovskite-yes" />
        ) : (
          <AlertTriangle size={14} className="opt-perovskite-no" />
        )}
        <span>
          Perovskite Confidence:{" "}
          <strong>{desc.perovskite_confidence.toFixed(0)}%</strong>
        </span>
      </div>
      <div className="opt-struct-section">
        <h4>Goldschmidt Criteria</h4>
        <div className="opt-struct-metrics">
          <div className="opt-struct-metric">
            <span className="opt-struct-metric-label">Tolerance Factor</span>
            <span className="opt-struct-metric-value">{desc.tolerance_factor.toFixed(4)}</span>
            <span className="opt-struct-metric-range">ideal: 0.88–1.05</span>
          </div>
          <div className="opt-struct-metric">
            <span className="opt-struct-metric-label">Octahedral Factor</span>
            <span className="opt-struct-metric-value">{desc.octahedral_factor.toFixed(4)}</span>
            <span className="opt-struct-metric-range">ideal: 0.41–0.73</span>
          </div>
        </div>
      </div>
      <div className="opt-struct-section">
        <h4>Bond Valence</h4>
        <div className="opt-struct-metrics">
          <Metric label="A-site BVS" value={desc.avg_bond_valence_a} digits={2} />
          <Metric label="B-site BVS" value={desc.avg_bond_valence_b} digits={2} />
          <Metric label="Mismatch" value={desc.bond_valence_mismatch} digits={3} />
        </div>
      </div>
      <div className="opt-struct-section">
        <h4>Site Classification</h4>
        <div className="opt-struct-sites">
          {Object.keys(desc.a_site_elements).length > 0 && (
            <div className="opt-struct-site">
              <span className="opt-site-label">A-site:</span>
              {Object.entries(desc.a_site_elements).map(([el, amt]) => (
                <span key={el} className="opt-element-tag opt-a-site">{el} ({amt.toFixed(2)})</span>
              ))}
            </div>
          )}
          {Object.keys(desc.b_site_elements).length > 0 && (
            <div className="opt-struct-site">
              <span className="opt-site-label">B-site:</span>
              {Object.entries(desc.b_site_elements).map(([el, amt]) => (
                <span key={el} className="opt-element-tag opt-b-site">{el} ({amt.toFixed(2)})</span>
              ))}
            </div>
          )}
          {Object.keys(desc.dopant_elements).length > 0 && (
            <div className="opt-struct-site">
              <span className="opt-site-label">Dopants:</span>
              {Object.entries(desc.dopant_elements).map(([el, amt]) => (
                <span key={el} className="opt-element-tag opt-dopant">{el} ({amt.toFixed(3)})</span>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="opt-struct-section">
        <h4>Physics Descriptors</h4>
        <div className="opt-struct-metrics opt-struct-physics">
          <Metric label="Avg EN" value={desc.avg_electronegativity} digits={3} />
          <Metric label="ΔEN (A-B)" value={desc.electronegativity_diff} digits={3} />
          <Metric label="Avg Mass" value={desc.avg_atomic_mass} digits={1} unit="amu" />
          <Metric label="Polarizability" value={desc.polarizability_index} digits={3} unit="ų" />
          <Metric label="A-site r" value={desc.avg_ionic_radius_a} digits={1} unit="pm" />
          <Metric label="B-site r" value={desc.avg_ionic_radius_b} digits={1} unit="pm" />
        </div>
      </div>
      {desc.warnings.length > 0 && (
        <div className="opt-struct-warnings">
          {desc.warnings.map((w, i) => (
            <span key={i} className="opt-struct-warning"><AlertTriangle size={12} /> {w}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, digits = 2, unit }: { label: string; value: number; digits?: number; unit?: string }) {
  return (
    <div className="opt-struct-metric">
      <span className="opt-struct-metric-label">{label}</span>
      <span className="opt-struct-metric-value">
        {value.toFixed(digits)}
        {unit && <span className="opt-struct-metric-unit">{unit}</span>}
      </span>
    </div>
  );
}

export default function StructuralAnalysis() {
  const {
    structuralResults, structuralLoading, structuralError,
    analyzeFormula, compareFormulas, clearStructural,
  } = useOptimizationStore();

  const [formula, setFormula] = useState("");
  const [compareMode, setCompareMode] = useState(false);
  const [compareList, setCompareList] = useState<string[]>([]);
  const [compareInput, setCompareInput] = useState("");

  const handleAnalyze = async () => {
    if (!formula.trim()) return;
    if (compareMode) {
      const all = [...compareList, formula.trim()].filter(Boolean);
      if (all.length > 0) await compareFormulas(all);
    } else {
      await analyzeFormula(formula.trim());
    }
  };

  const addToCompare = () => {
    if (compareInput.trim() && compareList.length < 10) {
      setCompareList([...compareList, compareInput.trim()]);
      setCompareInput("");
    }
  };

  return (
    <div className="opt-card opt-structural-card">
      <div className="opt-card-header">
        <Atom size={18} />
        <h3>Crystal Structure Analysis</h3>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginLeft: "auto" }}>
          <span style={{ fontSize: "12px", fontWeight: 500, color: "var(--text-secondary)" }}>
            Compare Mode
          </span>
          <button
            className={`toggle-switch ${compareMode ? "active" : ""}`}
            onClick={() => { setCompareMode(!compareMode); clearStructural(); }}
            role="switch"
            aria-checked={compareMode}
            aria-label="Toggle Compare Mode"
            style={{ transform: "scale(0.85)", transformOrigin: "right center" }}
          >
            <span className="toggle-knob" />
          </button>
        </div>
      </div>
      <p className="opt-card-description">
        Analyze structural properties using physics-based descriptors —
        tolerance factor, bond valence, and Goldschmidt criteria.
      </p>

      <div className="opt-struct-input-area">
        {compareMode ? (
          <div className="opt-compare-input">
            <div className="opt-compare-list">
              {compareList.map((f, i) => (
                <span key={i} className="opt-compare-chip">
                  {f}
                  <button onClick={() => setCompareList(compareList.filter((_, j) => j !== i))}>
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
            <div className="opt-compare-add">
              <FormulaValidationInput
                value={compareInput}
                onChange={setCompareInput}
                placeholder="Add formula to compare…"
                showDetails={false}
              />
              <button className="opt-compare-add-btn" onClick={addToCompare} disabled={!compareInput.trim()} title="Add formula to compare">
                <Plus size={18} />
              </button>
            </div>
            <button
              className="opt-analyze-btn"
              onClick={() => compareFormulas(compareList)}
              disabled={structuralLoading || compareList.length < 2}
            >
              {structuralLoading ? "Analyzing…" : `Compare ${compareList.length} formulas`}
            </button>
          </div>
        ) : (
          <div className="opt-single-input">
            <FormulaValidationInput
              value={formula}
              onChange={setFormula}
              placeholder="Enter formula (e.g. K0.5Na0.5NbO3)"
              showDetails={true}
            />
            <button
              className="opt-analyze-btn"
              onClick={handleAnalyze}
              disabled={structuralLoading || !formula.trim()}
            >
              {structuralLoading ? (
                <span className="opt-spinner" />
              ) : (
                <Search size={14} />
              )}
              Analyze
            </button>
          </div>
        )}
      </div>


      {structuralError && (
        <div className="opt-struct-error-banner">
          <AlertTriangle size={14} />
          {structuralError}
        </div>
      )}

      {structuralResults.length > 0 && (
        <div className="opt-struct-results">
          <div className="opt-struct-results-header">
            <span>{structuralResults.length} result(s)</span>
            <button className="opt-clear-btn" onClick={clearStructural}>
              Clear
            </button>
          </div>
          <div className="opt-struct-grid">
            {structuralResults.map((desc, i) => (
              <DescriptorCard key={`${desc.formula}-${i}`} desc={desc} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
