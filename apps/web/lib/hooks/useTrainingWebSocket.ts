/**
 * Piezo.AI — Training WebSocket Hook
 * =====================================
 * Connects to the training log stream and dispatches messages to the store.
 */

"use client";

import { useEffect, useRef } from "react";
import { APP_CONFIG } from "@/lib/constants";
import { useTrainingStore } from "@/lib/store/trainingStore";

export function useTrainingWebSocket(jobId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const addLog = useTrainingStore((s) => s.addLog);
  const updateProgress = useTrainingStore((s) => s.updateProgress);
  const addConvergencePoint = useTrainingStore((s) => s.addConvergencePoint);
  const setJobPhase = useTrainingStore((s) => s.setJobPhase);
  const setTrainedModels = useTrainingStore((s) => s.setTrainedModels);

  useEffect(() => {
    if (!jobId) return;

    const url = `${APP_CONFIG.api.wsUrl}/api/v1/training/ws/${jobId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog({
        type: "log",
        level: "info",
        message: "Connected to training stream...",
        timestamp: new Date().toISOString(),
      });
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
          case "log":
            addLog(msg);
            break;

          case "progress":
            updateProgress(msg.pct, msg.stage);
            addLog({ ...msg, type: "log", level: "info", message: `[${Math.round(msg.pct)}%] ${msg.stage}` });
            break;

          case "convergence":
            addConvergencePoint(msg.target, {
              iteration: msg.iteration,
              metric: msg.metric,
            });
            break;

          case "complete":
            setJobPhase("completed");
            if (msg.results?.models) {
              // Normalize worker format → TrainedModelInfo type
              // Worker sends: r2, rmse, n_train, n_test
              // UI expects: r2_score, rmse, n_train_samples, n_test_samples
              const normalized = msg.results.models.map((m: Record<string, unknown>) => ({
                id: m.id || "",
                display_name: m.display_name || `${m.algorithm}_${m.target}`,
                target: m.target,
                algorithm: m.algorithm,
                r2_score: m.r2_score ?? m.r2 ?? 0,
                rmse: m.rmse ?? 0,
                hyperparameters: m.hyperparameters || {},
                feature_dim: m.feature_dim ?? 0,
                n_train_samples: m.n_train_samples ?? m.n_train ?? 0,
                n_test_samples: m.n_test_samples ?? m.n_test ?? 0,
                model_file_path: m.model_file_path ?? m.model_path ?? "",
                training_duration_s: m.training_duration_s ?? 0,
                is_default: m.is_default ?? false,
                created_at: m.created_at || new Date().toISOString(),
              }));
              setTrainedModels(normalized);
            }
            addLog({
              type: "log",
              level: "success",
              message: "✅ Training complete!",
              timestamp: new Date().toISOString(),
            });
            break;

          case "cancelled":
            setJobPhase("cancelled");
            addLog({
              type: "log",
              level: "warning",
              message: "⚠️ Training cancelled by user",
              timestamp: new Date().toISOString(),
            });
            break;

          case "error":
            setJobPhase("failed");
            addLog({
              type: "log",
              level: "error",
              message: `❌ ${msg.message}`,
              timestamp: new Date().toISOString(),
            });
            break;
        }
      } catch {
        /* ignore malformed messages */
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    ws.onerror = () => {
      addLog({
        type: "log",
        level: "error",
        message: "WebSocket connection error",
        timestamp: new Date().toISOString(),
      });
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [jobId, addLog, updateProgress, addConvergencePoint, setJobPhase, setTrainedModels]);
}
