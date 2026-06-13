/**
 * Piezo.AI — Training Store (Zustand)
 * =====================================
 * State management for the training pipeline UI.
 */

import { create } from "zustand";
import type {
  AlgorithmInfo,
  TrainingJob,
  TrainedModelInfo,
  FieldIssue,
} from "@/lib/api/training";

/* ---------- Types ---------- */

export interface LogEntry {
  type: "log" | "progress" | "convergence" | "complete" | "error" | "cancelled";
  level?: "info" | "warning" | "error" | "success";
  message?: string;
  timestamp?: string;
  pct?: number;
  stage?: string;
  target?: string;
  iteration?: number;
  metric?: number;
}

export interface ConvergencePoint {
  iteration: number;
  metric: number;
}

export type ButtonState = "hidden" | "play" | "stop" | "checkmark";

export type JobPhase = "idle" | "configuring" | "validating" | "training" | "completed" | "failed" | "cancelled";

interface TrainingState {
  // ----- Pipeline config -----
  selectedDatasetId: string | null;
  selectedFields: string[];
  targets: string[];
  algorithms: Record<string, string>;
  hyperparameters: Record<string, Record<string, number | string>>;
  missingStrategies: Record<string, string>;
  mode: "manual" | "auto";

  // ----- Algorithm metadata -----
  algorithmList: AlgorithmInfo[];

  // ----- Validation -----
  validationIssues: FieldIssue[];

  // ----- Job state -----
  activeJobId: string | null;
  jobPhase: JobPhase;
  progress: number;
  currentStage: string;
  buttonState: ButtonState;

  // ----- Logs + convergence -----
  logs: LogEntry[];
  convergenceData: Record<string, ConvergencePoint[]>;

  // ----- Results -----
  trainedModels: TrainedModelInfo[];
  jobHistory: TrainingJob[];

  // ----- Actions -----
  setDataset: (id: string | null) => void;
  setSelectedFields: (fields: string[]) => void;
  setTargets: (targets: string[]) => void;
  setAlgorithm: (target: string, algo: string) => void;
  setUnifiedAlgorithm: (algo: string) => void;
  setHyperparam: (target: string, param: string, value: number | string) => void;
  setMissingStrategy: (field: string, strategy: string) => void;
  setMode: (mode: "manual" | "auto") => void;
  setAlgorithmList: (list: AlgorithmInfo[]) => void;
  setValidationIssues: (issues: FieldIssue[]) => void;
  setActiveJob: (jobId: string | null) => void;
  setJobPhase: (phase: JobPhase) => void;
  setButtonState: (state: ButtonState) => void;
  addLog: (entry: LogEntry) => void;
  updateProgress: (pct: number, stage: string) => void;
  addConvergencePoint: (target: string, point: ConvergencePoint) => void;
  setTrainedModels: (models: TrainedModelInfo[]) => void;
  setJobHistory: (jobs: TrainingJob[]) => void;
  resetPipeline: () => void;
  clearLogs: () => void;
}

const INITIAL_STATE = {
  selectedDatasetId: null as string | null,
  selectedFields: [] as string[],
  targets: [] as string[],
  algorithms: {} as Record<string, string>,
  hyperparameters: {} as Record<string, Record<string, number | string>>,
  missingStrategies: {} as Record<string, string>,
  mode: "manual" as const,
  algorithmList: [] as AlgorithmInfo[],
  validationIssues: [] as FieldIssue[],
  activeJobId: null as string | null,
  jobPhase: "idle" as JobPhase,
  progress: 0,
  currentStage: "",
  buttonState: "hidden" as ButtonState,
  logs: [] as LogEntry[],
  convergenceData: {} as Record<string, ConvergencePoint[]>,
  trainedModels: [] as TrainedModelInfo[],
  jobHistory: [] as TrainingJob[],
};

export const useTrainingStore = create<TrainingState>((set) => ({
  ...INITIAL_STATE,

  setDataset: (id) =>
    set({ selectedDatasetId: id, buttonState: id ? "play" : "hidden" }),

  setSelectedFields: (fields) => set({ selectedFields: fields }),
  setTargets: (targets) => set({ targets }),

  setAlgorithm: (target, algo) =>
    set((s) => ({ algorithms: { ...s.algorithms, [target]: algo } })),

  setUnifiedAlgorithm: (algo) =>
    set((s) => {
      const algs: Record<string, string> = {};
      for (const t of s.targets) algs[t] = algo;
      return { algorithms: algs };
    }),

  setHyperparam: (target, param, value) =>
    set((s) => ({
      hyperparameters: {
        ...s.hyperparameters,
        [target]: { ...s.hyperparameters[target], [param]: value },
      },
    })),

  setMissingStrategy: (field, strategy) =>
    set((s) => ({
      missingStrategies: { ...s.missingStrategies, [field]: strategy },
    })),

  setMode: (mode) => set({ mode }),
  setAlgorithmList: (list) => set({ algorithmList: list }),
  setValidationIssues: (issues) => set({ validationIssues: issues }),
  setActiveJob: (jobId) => set({ activeJobId: jobId }),

  setJobPhase: (phase) => {
    const buttonMap: Record<JobPhase, ButtonState> = {
      idle: "hidden",
      configuring: "play",
      validating: "play",
      training: "stop",
      completed: "checkmark",
      failed: "play",
      cancelled: "play",
    };
    set({ jobPhase: phase, buttonState: buttonMap[phase] });
  },

  setButtonState: (state) => set({ buttonState: state }),

  addLog: (entry) =>
    set((s) => ({ logs: [...s.logs, entry] })),

  updateProgress: (pct, stage) =>
    set({ progress: pct, currentStage: stage }),

  addConvergencePoint: (target, point) =>
    set((s) => ({
      convergenceData: {
        ...s.convergenceData,
        [target]: [...(s.convergenceData[target] || []), point],
      },
    })),

  setTrainedModels: (models) => set({ trainedModels: models }),
  setJobHistory: (jobs) => set({ jobHistory: jobs }),

  resetPipeline: () =>
    set({
      ...INITIAL_STATE,
    }),

  clearLogs: () =>
    set({ logs: [], convergenceData: {}, progress: 0, currentStage: "" }),
}));
