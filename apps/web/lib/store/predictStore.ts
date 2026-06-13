/**
 * Piezo.AI — Predict Store (Zustand)
 * =====================================
 * State management for the prediction UI.
 * Supports per-target model selection (d33, tc, hardness independently).
 */

import { create } from "zustand";
import type {
  PredictResponse,
  TrainedModelItem,
  FormulaValidation,
  CompositeParams,
  BatchPredictSummary,
} from "@/lib/api/predictions";

/* ---------- Types ---------- */

export interface ComparisonEntry {
  id: string;
  formula: string;
  prediction: PredictResponse;
  timestamp: string;
}

export type PredictTab = "single" | "batch" | "comparison";

/** Per-target model selection: user picks one model per property (or null = skip) */
export interface TargetModelSelection {
  d33: string | null;
  tc: string | null;
  vickers_hardness: string | null;
}

interface PredictState {
  // ----- Tab -----
  activeTab: PredictTab;

  // ----- Model selection (per-target) -----
  models: TrainedModelItem[];
  targetModels: TargetModelSelection;
  /** Legacy: single model id for batch prediction */
  selectedModelId: string | null;
  modelsLoading: boolean;

  // ----- Formula input -----
  formula: string;
  formulaValidation: FormulaValidation | null;
  formulaValidating: boolean;

  // ----- Composite -----
  isComposite: boolean;
  compositeParams: CompositeParams;

  // ----- Prediction -----
  prediction: PredictResponse | null;
  predicting: boolean;
  predictionError: string | null;

  // ----- Comparison -----
  comparisonList: ComparisonEntry[];
  /** Temporary "Added!" confirmation flag */
  comparisonJustAdded: boolean;

  // ----- Batch -----
  batchResult: BatchPredictSummary | null;
  batchLoading: boolean;
  batchError: string | null;

  // ----- Actions -----
  setActiveTab: (tab: PredictTab) => void;
  setModels: (models: TrainedModelItem[]) => void;
  setTargetModel: (target: keyof TargetModelSelection, modelId: string | null) => void;
  setSelectedModel: (id: string | null) => void;
  setModelsLoading: (loading: boolean) => void;
  setFormula: (formula: string) => void;
  setFormulaValidation: (v: FormulaValidation | null) => void;
  setFormulaValidating: (v: boolean) => void;
  setIsComposite: (v: boolean) => void;
  setCompositeParam: (key: keyof CompositeParams, value: string | number | undefined) => void;
  resetCompositeParams: () => void;
  setPrediction: (p: PredictResponse | null) => void;
  setPredicting: (v: boolean) => void;
  setPredictionError: (e: string | null) => void;
  addToComparison: (entry: ComparisonEntry) => void;
  removeFromComparison: (id: string) => void;
  clearComparison: () => void;
  setComparisonJustAdded: (v: boolean) => void;
  setBatchResult: (r: BatchPredictSummary | null) => void;
  setBatchLoading: (v: boolean) => void;
  setBatchError: (e: string | null) => void;
  reset: () => void;
}

const DEFAULT_COMPOSITE: CompositeParams = {
  matrix_type: "none",
  filler_wt_pct: 0,
  particle_morphology: "none",
  particle_size_nm: undefined,
  surface_treatment: "none",
  fabrication_method: "conventional",
};

const INITIAL_STATE = {
  activeTab: "single" as PredictTab,
  models: [] as TrainedModelItem[],
  targetModels: { d33: null, tc: null, vickers_hardness: null } as TargetModelSelection,
  selectedModelId: null as string | null,
  modelsLoading: false,
  formula: "",
  formulaValidation: null as FormulaValidation | null,
  formulaValidating: false,
  isComposite: false,
  compositeParams: { ...DEFAULT_COMPOSITE },
  prediction: null as PredictResponse | null,
  predicting: false,
  predictionError: null as string | null,
  comparisonList: [] as ComparisonEntry[],
  comparisonJustAdded: false,
  batchResult: null as BatchPredictSummary | null,
  batchLoading: false,
  batchError: null as string | null,
};

export const usePredictStore = create<PredictState>((set) => ({
  ...INITIAL_STATE,

  setActiveTab: (tab) => set({ activeTab: tab }),
  setModels: (models) => set({ models }),
  setTargetModel: (target, modelId) =>
    set((s) => ({
      targetModels: { ...s.targetModels, [target]: modelId },
    })),
  setSelectedModel: (id) => set({ selectedModelId: id }),
  setModelsLoading: (loading) => set({ modelsLoading: loading }),
  setFormula: (formula) => set({ formula }),
  setFormulaValidation: (v) => set({ formulaValidation: v }),
  setFormulaValidating: (v) => set({ formulaValidating: v }),
  setIsComposite: (v) =>
    set((s) => ({
      isComposite: v,
      compositeParams: v ? s.compositeParams : { ...DEFAULT_COMPOSITE },
    })),
  setCompositeParam: (key, value) =>
    set((s) => ({
      compositeParams: { ...s.compositeParams, [key]: value },
    })),
  resetCompositeParams: () => set({ compositeParams: { ...DEFAULT_COMPOSITE } }),
  setPrediction: (p) => set({ prediction: p }),
  setPredicting: (v) => set({ predicting: v }),
  setPredictionError: (e) => set({ predictionError: e }),
  addToComparison: (entry) =>
    set((s) => ({
      comparisonList: [...s.comparisonList, entry],
      comparisonJustAdded: true,
    })),
  removeFromComparison: (id) =>
    set((s) => ({
      comparisonList: s.comparisonList.filter((e) => e.id !== id),
    })),
  clearComparison: () => set({ comparisonList: [] }),
  setComparisonJustAdded: (v) => set({ comparisonJustAdded: v }),
  setBatchResult: (r) => set({ batchResult: r }),
  setBatchLoading: (v) => set({ batchLoading: v }),
  setBatchError: (e) => set({ batchError: e }),
  reset: () => set({ ...INITIAL_STATE }),
}));
