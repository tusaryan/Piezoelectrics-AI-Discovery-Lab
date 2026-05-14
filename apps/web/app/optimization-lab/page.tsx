"use client";

import { FlaskConical, Atom, Target } from "lucide-react";
import { useState } from "react";
import OptModelSelector from "@/components/optimization/ModelSelector";
import OptimizationConfigPanel from "@/components/optimization/OptimizationConfig";
import ParetoChart from "@/components/optimization/ParetoChart";
import SolutionTable from "@/components/optimization/SolutionTable";
import OptConvergenceChart from "@/components/optimization/ConvergenceChart";
import StructuralAnalysis from "@/components/optimization/StructuralAnalysis";
import { useOptimizationStore } from "@/lib/store/optimizationStore";

type TabKey = "optimization" | "structural";

export default function OptimizationLabPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("optimization");
  const { solutions, optimizationError } = useOptimizationStore();

  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <FlaskConical size={22} />
        </div>
        <div className="page-header-text">
          <h1>Optimization Lab</h1>
          <p>
            Crystal structure analysis + multi-objective property optimization
          </p>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="opt-tabs">
        <button
          className={`opt-tab ${activeTab === "optimization" ? "active" : ""}`}
          onClick={() => setActiveTab("optimization")}
        >
          <Target size={16} />
          Property Optimization
        </button>
        <button
          className={`opt-tab ${activeTab === "structural" ? "active" : ""}`}
          onClick={() => setActiveTab("structural")}
        >
          <Atom size={16} />
          Structure Analysis
        </button>
      </div>

      {/* Tab content */}
      {activeTab === "optimization" && (
        <div className="opt-layout">
          {/* Left column: config */}
          <div className="opt-config-col">
            <OptModelSelector />
            <OptimizationConfigPanel />
          </div>

          {/* Right column: results */}
          <div className="opt-results-col">
            {optimizationError && (
              <div className="opt-error-banner">
                <span>⚠️ {optimizationError}</span>
              </div>
            )}

            {solutions.length > 0 ? (
              <>
                <ParetoChart />
                <OptConvergenceChart />
                <SolutionTable />
              </>
            ) : (
              <div className="opt-empty-results">
                <FlaskConical size={48} />
                <h3>No Optimization Results Yet</h3>
                <p>
                  Select models, configure objectives, and click{" "}
                  <strong>Run Optimization</strong> to find Pareto-optimal
                  compositions.
                </p>
                <div className="opt-empty-steps">
                  <div className="opt-step">
                    <span className="opt-step-num">1</span>
                    Select surrogate models for each target property
                  </div>
                  <div className="opt-step">
                    <span className="opt-step-num">2</span>
                    Choose a use-case preset or configure custom objectives
                  </div>
                  <div className="opt-step">
                    <span className="opt-step-num">3</span>
                    Run NSGA-II to find Pareto-optimal compositions
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "structural" && (
        <div className="opt-structural-layout">
          <StructuralAnalysis />
        </div>
      )}
    </div>
  );
}
