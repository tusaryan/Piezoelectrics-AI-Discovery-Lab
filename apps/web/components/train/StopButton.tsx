/**
 * StopButton — state machine: hidden → play → stop → checkmark
 */

"use client";

import { Play, Square, CheckCircle2, Loader2 } from "lucide-react";
import { useTrainingStore, type ButtonState } from "@/lib/store/trainingStore";
import { createTrainingJob, stopTrainingJob } from "@/lib/api/training";
import { useState } from "react";

export default function StopButton() {
  const {
    buttonState, selectedDatasetId, targets, algorithms,
    hyperparameters, selectedFields, missingStrategies, mode,
    setActiveJob, setJobPhase, clearLogs, setButtonState,
  } = useTrainingStore();

  const [loading, setLoading] = useState(false);
  const activeJobId = useTrainingStore((s) => s.activeJobId);

  if (buttonState === "hidden") return null;

  const handlePlay = async () => {
    if (!selectedDatasetId || targets.length === 0) return;

    // Ensure all targets have an algorithm
    const algoCheck = targets.every((t) => algorithms[t]);
    if (!algoCheck) return;

    setLoading(true);
    clearLogs();

    try {
      const job = await createTrainingJob({
        dataset_id: selectedDatasetId,
        targets,
        algorithms,
        hyperparameters: hyperparameters as Record<string, Record<string, number>>,
        selected_fields: selectedFields,
        missing_strategies: missingStrategies,
        mode,
      });
      setActiveJob(job.id);
      setJobPhase("training");
    } catch (err) {
      console.error("Failed to start training:", err);
      setJobPhase("failed");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!activeJobId) return;
    setLoading(true);
    try {
      await stopTrainingJob(activeJobId);
      setJobPhase("cancelled");
    } catch (err) {
      // If job already finished/not running, treat as completed gracefully
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("not running") || msg.includes("already finished")) {
        // Job completed before stop signal arrived — show checkmark
        setJobPhase("completed");
      } else {
        console.error("Failed to stop training:", msg);
        // Still transition to avoid stuck UI
        setJobPhase("failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    clearLogs();
    setActiveJob(null);
    setJobPhase("configuring");
  };

  const configs: Record<ButtonState, {
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
    className: string;
  }> = {
    hidden: { icon: null, label: "", onClick: () => {}, className: "" },
    play: {
      icon: loading ? <Loader2 size={18} className="spin" /> : <Play size={18} />,
      label: loading ? "Starting..." : "Start Training",
      onClick: handlePlay,
      className: "stop-btn play",
    },
    stop: {
      icon: loading ? <Loader2 size={18} className="spin" /> : <Square size={18} />,
      label: loading ? "Stopping..." : "Stop Training",
      onClick: handleStop,
      className: "stop-btn stop",
    },
    checkmark: {
      icon: <CheckCircle2 size={18} />,
      label: "Complete — Train Again",
      onClick: handleReset,
      className: "stop-btn checkmark",
    },
  };

  const config = configs[buttonState];

  return (
    <button
      className={config.className}
      onClick={config.onClick}
      disabled={loading}
    >
      {config.icon}
      <span>{config.label}</span>
    </button>
  );
}
