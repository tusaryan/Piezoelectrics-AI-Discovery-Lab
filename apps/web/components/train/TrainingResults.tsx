/**
 * TrainingResults — Metrics cards + Predicted vs Actual scatter after training.
 */

"use client";

import { Award, Clock, Database, TrendingUp } from "lucide-react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useTrainingStore } from "@/lib/store/trainingStore";

function getR2Color(r2: number): string {
  if (r2 >= 0.8) return "var(--success)";
  if (r2 >= 0.5) return "var(--warning)";
  return "var(--error)";
}

export default function TrainingResults() {
  const trainedModels = useTrainingStore((s) => s.trainedModels);
  const jobPhase = useTrainingStore((s) => s.jobPhase);

  if (jobPhase !== "completed" || trainedModels.length === 0) return null;

  return (
    <div className="training-results">
      <h3 className="results-title">
        <Award size={18} /> Training Results
      </h3>

      {/* Metrics Cards */}
      <div className="results-metrics-grid">
        {trainedModels.map((model) => (
          <div key={`${model.target}-${model.algorithm}`} className="result-card">
            <div className="result-card-header">
              <span className="result-target">{model.target}</span>
              <span className="result-algo">{model.algorithm}</span>
            </div>

            <div className="result-card-metrics">
              <div className="result-metric">
                <TrendingUp size={14} />
                <span className="result-metric-label">R²</span>
                <span
                  className="result-metric-value"
                  style={{ color: getR2Color(model.r2_score ?? 0) }}
                >
                  {(model.r2_score ?? 0).toFixed(4)}
                </span>
              </div>
              <div className="result-metric">
                <TrendingUp size={14} />
                <span className="result-metric-label">RMSE</span>
                <span className="result-metric-value">
                  {(model.rmse ?? 0).toFixed(2)}
                </span>
              </div>
            </div>

            <div className="result-card-footer">
              <span><Clock size={12} /> {(model.training_duration_s ?? 0).toFixed(1)}s</span>
              <span><Database size={12} /> {model.n_train_samples}/{model.n_test_samples}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
