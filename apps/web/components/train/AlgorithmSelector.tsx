/**
 * AlgorithmSelector — card grid for selecting ML algorithms.
 *
 * Per-Target mode:
 *   - Targets are shown as clickable highlight boxes
 *   - First target is highlighted initially
 *   - Clicking an algorithm assigns it to the highlighted target, then auto-advances
 *   - User can click any target box to re-highlight it and reassign its algorithm
 */

"use client";

import { useState, useEffect } from "react";
import {
  TreePine, Network, CircleDot, Zap, BarChart3,
  GitBranch, Brain, Layers,
} from "lucide-react";
import { useTrainingStore } from "@/lib/store/trainingStore";

const ALGO_ICONS: Record<string, React.ElementType> = {
  xgboost: Zap,
  random_forest: TreePine,
  svr: CircleDot,
  lightgbm: BarChart3,
  gradient_boosting: GitBranch,
  decision_tree: TreePine,
  ann: Brain,
  stacking: Layers,
};

export default function AlgorithmSelector() {
  const {
    targets, algorithms, algorithmList, mode,
    setAlgorithm, setUnifiedAlgorithm, setMode,
  } = useTrainingStore();

  const [unified, setUnified] = useState(true);
  // Index of the currently highlighted target in per-target mode
  const [activeTargetIdx, setActiveTargetIdx] = useState(0);

  // Reset highlight when mode changes or targets change
  useEffect(() => {
    setActiveTargetIdx(0);
  }, [unified, targets.length]);

  const handleAlgoClick = (algoKey: string) => {
    if (unified) {
      setUnifiedAlgorithm(algoKey);
    } else if (targets.length > 0) {
      // Assign to the currently highlighted target
      const target = targets[activeTargetIdx] || targets[0];
      setAlgorithm(target, algoKey);

      // Auto-advance to next target (wrap around to stay at last)
      if (activeTargetIdx < targets.length - 1) {
        setActiveTargetIdx(activeTargetIdx + 1);
      }
    }
  };

  // Current selected algorithm(s)
  const selectedAlgos = new Set(Object.values(algorithms));

  // In per-target mode, also highlight the algo card for the active target
  const activeTargetAlgo = !unified && targets.length > 0
    ? algorithms[targets[activeTargetIdx]]
    : null;

  return (
    <div className="algorithm-selector">
      <div className="algo-header">
        <h3 className="config-section-title">
          <Brain size={16} /> Algorithm Selection
        </h3>
        <div className="algo-mode-toggle">
          <button
            className={`algo-mode-btn ${unified ? "active" : ""}`}
            onClick={() => setUnified(true)}
          >
            Unified
          </button>
          <button
            className={`algo-mode-btn ${!unified ? "active" : ""}`}
            onClick={() => setUnified(false)}
          >
            Per-Target
          </button>
          <span className="algo-divider" />
          <button
            className={`algo-mode-btn ${mode === "auto" ? "active auto" : ""}`}
            onClick={() => setMode(mode === "auto" ? "manual" : "auto")}
            title="Auto-tune with Optuna"
          >
            {mode === "auto" ? "⚡ Auto-Tune ON" : "Auto-Tune"}
          </button>
        </div>
      </div>

      {/* Per-Target: clickable target boxes with highlight */}
      {!unified && targets.length > 0 && (
        <div className="per-target-selectors">
          {targets.map((target, idx) => (
            <button
              key={target}
              className={`per-target-box ${idx === activeTargetIdx ? "highlighted" : ""} ${algorithms[target] ? "assigned" : ""}`}
              onClick={() => setActiveTargetIdx(idx)}
            >
              <span className="per-target-label">{target}</span>
              {algorithms[target] ? (
                <span className="per-target-algo-badge">
                  {algorithmList.find((a) => a.key === algorithms[target])?.display_name || algorithms[target]}
                </span>
              ) : (
                <span className="per-target-placeholder">Click an algorithm below ↓</span>
              )}
            </button>
          ))}
          <p className="per-target-hint">
            Selecting: <strong>{targets[activeTargetIdx]}</strong> — click any target box to switch
          </p>
        </div>
      )}

      <div className="algo-grid">
        {algorithmList.map((algo) => {
          const Icon = ALGO_ICONS[algo.key] || Brain;
          const isSelected = unified
            ? selectedAlgos.has(algo.key)
            : algo.key === activeTargetAlgo;

          return (
            <button
              key={algo.key}
              className={`algo-card ${isSelected ? "selected" : ""}`}
              onClick={() => handleAlgoClick(algo.key)}
            >
              <div className="algo-card-icon">
                <Icon size={20} />
              </div>
              <div className="algo-card-content">
                <h4>{algo.display_name}</h4>
                <p>{algo.description}</p>
                {algo.supports_convergence && (
                  <span className="algo-badge">Convergence</span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
