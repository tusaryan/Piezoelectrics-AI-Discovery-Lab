/**
 * ProgressBar — ML-stage weighted progress bar with stage label.
 */

"use client";

import { useTrainingStore } from "@/lib/store/trainingStore";

export default function ProgressBar() {
  const progress = useTrainingStore((s) => s.progress);
  const currentStage = useTrainingStore((s) => s.currentStage);
  const jobPhase = useTrainingStore((s) => s.jobPhase);

  if (jobPhase === "idle" || jobPhase === "configuring") return null;

  const pct = Math.min(100, Math.max(0, progress));
  const displayPct = Math.round(pct);

  return (
    <div className="training-progress">
      <div className="progress-header">
        <span className="progress-stage">{currentStage || "Initializing..."}</span>
        <span className="progress-pct">{displayPct}%</span>
      </div>
      <div className="progress-track">
        <div
          className={`progress-fill ${jobPhase === "completed" ? "complete" : ""} ${jobPhase === "failed" ? "error" : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
