/**
 * TrainPageContent — main orchestrator for the Model Studio page.
 * Two-panel layout: config (left) + monitor (right).
 */

"use client";

import { useEffect } from "react";
import { BrainCircuit } from "lucide-react";
import PipelineConfigurator from "./PipelineConfigurator";
import AlgorithmSelector from "./AlgorithmSelector";
import HyperparameterPanel from "./HyperparameterPanel";
import TrainingTerminal from "./TrainingTerminal";
import ProgressBar from "./ProgressBar";
import StopButton from "./StopButton";
import ConvergenceChart from "./ConvergenceChart";
import TrainingResults from "./TrainingResults";
import { useTrainingStore } from "@/lib/store/trainingStore";
import { useTrainingWebSocket } from "@/lib/hooks/useTrainingWebSocket";
import { getAlgorithms } from "@/lib/api/training";

export default function TrainPageContent() {
  const { algorithmList, setAlgorithmList, activeJobId, jobPhase } = useTrainingStore();

  // Load algorithm list on mount
  useEffect(() => {
    if (algorithmList.length === 0) {
      getAlgorithms()
        .then(setAlgorithmList)
        .catch((err) => console.error("Failed to load algorithms:", err));
    }
  }, []);

  // Connect WebSocket for active job
  useTrainingWebSocket(activeJobId);

  const isTraining = jobPhase === "training";
  const showMonitor = jobPhase !== "idle";

  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <BrainCircuit size={22} />
        </div>
        <div className="page-header-text">
          <h1>Model Studio</h1>
          <p>Configure ML pipelines, train models, view convergence and results</p>
        </div>
      </div>

      <div className="train-layout">
        {/* Left Panel — Configuration */}
        <div className={`train-panel-config ${isTraining ? "training-active" : ""}`}>
          <PipelineConfigurator />
          {algorithmList.length > 0 && <AlgorithmSelector />}
          <HyperparameterPanel />
        </div>

        {/* Right Panel — Monitor */}
        <div className="train-panel-monitor">
          <div className="monitor-controls">
            <StopButton />
          </div>
          <ProgressBar />
          <TrainingTerminal />
          <ConvergenceChart />
          <TrainingResults />
        </div>
      </div>
    </div>
  );
}
