"use client";

/**
 * Symbolic Regression Card — PySR equations with KaTeX rendering.
 *
 * Shows discovered equations ranked by complexity/accuracy,
 * parsimony Pareto front, and best equation highlight.
 */

import { useEffect, useState, useRef } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Loader2, Sigma, Maximize2, Minimize2, Play, AlertCircle } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";
import InfoTooltip from "./InfoTooltip";

function KaTeXBlock({ latex }: { latex: string }) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    try {
      // Dynamic import to handle SSR
      import("katex").then((katex) => {
        if (ref.current) {
          katex.default.render(latex, ref.current, {
            throwOnError: false,
            displayMode: false,
            output: "html",
          });
        }
      });
    } catch {
      if (ref.current) ref.current.textContent = latex;
    }
  }, [latex]);

  return <span ref={ref} className="katex-container" />;
}

export default function SymbolicRegression() {
  const {
    symbolicRegression, symRegLoading, symRegError,
    selectedModelId, fetchSymbolicRegression,
    pysrInstalling, installPySRBackend,
  } = useInterpretStore();
  const [expanded, setExpanded] = useState(false);
  const [selectedEqIdx, setSelectedEqIdx] = useState<number | null>(null);

  const handleRun = () => {
    if (selectedModelId) fetchSymbolicRegression();
  };

  const handleInstall = () => {
    installPySRBackend();
  };

  const result = symbolicRegression;
  const equations = result?.equations ?? [];
  const pareto = result?.pareto_front ?? [];

  // Highlight best equation
  const bestIdx = equations.findIndex(
    (e) => result?.best_equation && e.equation_str === result.best_equation.equation_str
  );

  return (
    <div className={`interpret-card ${expanded ? "expanded" : ""}`} id="symbolic-regression">
      <div className="interpret-card-header">
        <div className="interpret-card-title">
          <Sigma size={16} />
          <span>Symbolic Regression (PySR)</span>
        </div>
        <div className="interpret-card-actions">
          <InfoTooltip title="Symbolic Regression">
            <p>Discovers interpretable mathematical equations that approximate the ML model.</p>
            <p><strong>How it works:</strong> PySR uses genetic programming to evolve equations, balancing accuracy vs complexity.</p>
            <p><strong>Pareto front:</strong> Shows the trade-off — simpler equations (left) vs more accurate ones (right).</p>
            <p><strong>Best equation:</strong> The equation with highest R² — the most accurate discovered formula.</p>
            <p>⚠️ Requires Julia backend. May take 1-3 minutes.</p>
          </InfoTooltip>
          <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      <div className="interpret-card-body">
        {/* Not available warning */}
        {result && !result.available && (
          <div className="symreg-unavailable">
            <AlertCircle size={18} />
            <div>
              <strong>PySR Backend Missing</strong>
              <p>The Symbolic Regression engine requires Julia to run.</p>
              <button 
                className="btn btn-primary btn-sm mt-2" 
                onClick={handleInstall}
                disabled={pysrInstalling}
                style={{ fontSize: 12, padding: "6px 12px", background: "var(--primary)", color: "white", borderRadius: "6px", border: "none", cursor: "pointer" }}
              >
                {pysrInstalling ? (
                  <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <Loader2 size={14} className="spin" /> Installing Julia Backend (this takes a few minutes)...
                  </span>
                ) : (
                  "Install Julia Backend Automatically"
                )}
              </button>
            </div>
          </div>
        )}

        {/* Run button */}
        {!result && !symRegLoading && (
          <div className="symreg-run-section">
            <p className="symreg-description">
              Discover interpretable mathematical equations relating material features to properties.
              This uses PySR (symbolic regression via Julia) and may take 1-3 minutes.
            </p>
            <button
              className="symreg-run-btn"
              onClick={handleRun}
              disabled={!selectedModelId}
            >
              <Play size={14} />
              Run Symbolic Regression
            </button>
          </div>
        )}

        {/* Loading */}
        {symRegLoading && (
          <div className="interpret-loading">
            <Loader2 size={20} className="spin" />
            <span>Running symbolic regression — this may take 1-3 minutes...</span>
          </div>
        )}

        {/* Error */}
        {symRegError && <div className="interpret-error">{symRegError}</div>}

        {/* Results */}
        {result && result.available && equations.length > 0 && !symRegLoading && (
          <div className="symreg-results">
            {/* Best equation highlight */}
            {result.best_equation && (
              <div className="symreg-best">
                <div className="symreg-best-label">Best Equation (R² = {result.best_equation.r2.toFixed(4)})</div>
                <div className="symreg-best-equation">
                  <KaTeXBlock latex={result.best_equation.latex} />
                </div>
                <div className="symreg-best-readable font-mono">
                  {result.best_equation.readable}
                </div>
              </div>
            )}

            {/* Pareto front chart */}
            {pareto.length > 1 && (
              <div className="symreg-pareto">
                <h4>Parsimony Pareto Front</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                    <XAxis
                      dataKey="complexity" type="number" name="Complexity"
                      tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                      label={{ value: "Complexity", position: "bottom", offset: 15, fill: "var(--text-secondary)", fontSize: 11 }}
                    />
                    <YAxis
                      dataKey="r2" type="number" name="R²"
                      tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
                      label={{ value: "R²", angle: -90, position: "insideLeft", fill: "var(--text-secondary)", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--card)", border: "1px solid var(--border)",
                        borderRadius: "8px", fontSize: "12px", color: "var(--text)",
                      }}
                    />
                    <Scatter data={pareto}>
                      {pareto.map((_, i) => (
                        <Cell
                          key={i}
                          fill={i === bestIdx ? "var(--success)" : "var(--chart-pareto)"}
                          r={i === bestIdx ? 6 : 4}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Equations table */}
            <div className="symreg-table">
              <h4>All Equations ({equations.length})</h4>
              <div className="symreg-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Equation</th>
                      <th>Complexity</th>
                      <th>R²</th>
                      <th>Loss</th>
                    </tr>
                  </thead>
                  <tbody>
                    {equations.map((eq, i) => (
                      <tr
                        key={i}
                        className={`${i === bestIdx ? "best-row" : ""} ${selectedEqIdx === i ? "selected" : ""}`}
                        onClick={() => setSelectedEqIdx(selectedEqIdx === i ? null : i)}
                      >
                        <td>{i + 1}</td>
                        <td className="eq-cell">
                          <KaTeXBlock latex={eq.latex} />
                        </td>
                        <td>{eq.complexity}</td>
                        <td>{eq.r2.toFixed(4)}</td>
                        <td>{eq.loss.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {selectedEqIdx !== null && equations[selectedEqIdx] && (
                <div className="symreg-selected-detail">
                  <div className="detail-label">Readable form:</div>
                  <code className="font-mono">{equations[selectedEqIdx].readable}</code>
                </div>
              )}
            </div>

            <div className="symreg-meta">
              {result.n_samples} samples • {result.n_features} features
            </div>
          </div>
        )}

        {result && result.available && equations.length === 0 && !symRegLoading && (
          <div className="interpret-empty">
            No equations discovered. Try increasing iterations or complexity.
          </div>
        )}
      </div>
    </div>
  );
}
