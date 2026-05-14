/**
 * Optimization Store — Zustand state management for Optimization Lab.
 */
import { create } from "zustand";
import type {
  OptimizationModel,
  StructuralDescriptor,
  ParetoSolution,
  ObjectiveConfig,
  UseCasePreset,
} from "@/lib/api/optimization";
import {
  fetchOptimizationModels,
  runStructuralAnalysis,
  runStructuralComparison,
  runOptimization,
  fetchPresets,
} from "@/lib/api/optimization";

interface OptimizationState {
  // Models
  models: OptimizationModel[];
  modelsLoading: boolean;
  selectedModelIds: Record<string, string>; // target -> model_id

  // Structural analysis
  structuralResults: StructuralDescriptor[];
  structuralLoading: boolean;
  structuralError: string | null;

  // Optimization
  presets: UseCasePreset[];
  activePreset: string;
  objectives: Record<string, ObjectiveConfig>;
  popSize: number;
  nGenerations: number;
  solutions: ParetoSolution[];
  convergence: Record<string, number>[];
  optimizationLoading: boolean;
  optimizationError: string | null;
  optimizationStats: {
    n_generations_run: number;
    n_evaluations: number;
    duration_seconds: number;
    targets_optimized: string[];
  } | null;

  // Actions
  fetchModels: () => Promise<void>;
  setSelectedModel: (target: string, modelId: string) => void;
  analyzeFormula: (formula: string) => Promise<void>;
  compareFormulas: (formulas: string[]) => Promise<void>;
  clearStructural: () => void;
  loadPresets: () => Promise<void>;
  setPreset: (key: string) => void;
  setObjective: (target: string, config: ObjectiveConfig) => void;
  setPopSize: (size: number) => void;
  setNGenerations: (n: number) => void;
  runOptimization: () => Promise<void>;
  clearOptimization: () => void;
}

const DEFAULT_OBJECTIVES: Record<string, ObjectiveConfig> = {
  d33: { direction: "maximize", min: 50, max: 700, weight: 1.0 },
  tc: { direction: "maximize", min: 100, max: 500, weight: 1.0 },
  vickers_hardness: { direction: "maximize", min: 50, max: 1200, weight: 1.0 },
};

/** Hardcoded presets so UI works even before API loads. */
const FALLBACK_PRESETS: UseCasePreset[] = [
  {
    key: "flexible_wearables",
    label: "🔋 Flexible Wearables",
    description: "High d33 + low hardness + moderate tc",
    objectives: {
      d33: { direction: "maximize", min: 200, max: 700, weight: 1.0 },
      tc: { direction: "maximize", min: 150, max: 350, weight: 0.5 },
      vickers_hardness: { direction: "minimize", min: 50, max: 300, weight: 0.7 },
    },
  },
  {
    key: "industrial_actuators",
    label: "⚡ Industrial Actuators",
    description: "Moderate d33 + high hardness + high tc",
    objectives: {
      d33: { direction: "maximize", min: 100, max: 500, weight: 0.7 },
      tc: { direction: "maximize", min: 300, max: 500, weight: 1.0 },
      vickers_hardness: { direction: "maximize", min: 300, max: 1200, weight: 0.8 },
    },
  },
  {
    key: "ultrasonic_transducers",
    label: "🔊 Ultrasonic Transducers",
    description: "Moderate d33 + very high hardness + high tc",
    objectives: {
      d33: { direction: "maximize", min: 80, max: 400, weight: 0.6 },
      tc: { direction: "maximize", min: 350, max: 500, weight: 0.9 },
      vickers_hardness: { direction: "maximize", min: 500, max: 1200, weight: 1.0 },
    },
  },
  {
    key: "custom",
    label: "🎯 Custom",
    description: "User-defined property ranges",
    objectives: {
      d33: { direction: "maximize", min: 50, max: 700, weight: 1.0 },
      tc: { direction: "maximize", min: 100, max: 500, weight: 1.0 },
      vickers_hardness: { direction: "maximize", min: 50, max: 1200, weight: 1.0 },
    },
  },
];

export const useOptimizationStore = create<OptimizationState>((set, get) => ({
  // State
  models: [],
  modelsLoading: false,
  selectedModelIds: {},
  structuralResults: [],
  structuralLoading: false,
  structuralError: null,
  presets: FALLBACK_PRESETS,
  activePreset: "custom",
  objectives: { ...DEFAULT_OBJECTIVES },
  popSize: 100,
  nGenerations: 50,
  solutions: [],
  convergence: [],
  optimizationLoading: false,
  optimizationError: null,
  optimizationStats: null,

  // Actions
  fetchModels: async () => {
    set({ modelsLoading: true });
    try {
      const models = await fetchOptimizationModels();
      // Auto-select default models
      const selected: Record<string, string> = {};
      for (const m of models) {
        if (m.is_default && !selected[m.target]) {
          selected[m.target] = m.id;
        }
      }
      // Fill remaining with first available
      for (const m of models) {
        if (!selected[m.target]) {
          selected[m.target] = m.id;
        }
      }
      set({ models, selectedModelIds: selected, modelsLoading: false });
    } catch {
      set({ modelsLoading: false });
    }
  },

  setSelectedModel: (target, modelId) => {
    set((s) => ({
      selectedModelIds: { ...s.selectedModelIds, [target]: modelId },
    }));
  },

  analyzeFormula: async (formula) => {
    set({ structuralLoading: true, structuralError: null });
    try {
      const result = await runStructuralAnalysis(formula);
      set((s) => ({
        structuralResults: [...s.structuralResults, result],
        structuralLoading: false,
      }));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Analysis failed";
      set({ structuralLoading: false, structuralError: msg });
    }
  },

  compareFormulas: async (formulas) => {
    set({ structuralLoading: true, structuralError: null });
    try {
      const results = await runStructuralComparison(formulas);
      set({ structuralResults: results, structuralLoading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Comparison failed";
      set({ structuralLoading: false, structuralError: msg });
    }
  },

  clearStructural: () => {
    set({ structuralResults: [], structuralError: null });
  },

  loadPresets: async () => {
    try {
      const presets = await fetchPresets();
      if (presets.length > 0) {
        set({ presets });
      }
    } catch {
      // Keep fallback presets — they're already set
    }
  },

  setPreset: (key) => {
    const { presets } = get();
    const preset = presets.find((p) => p.key === key)
      ?? FALLBACK_PRESETS.find((p) => p.key === key);
    if (preset) {
      set({ activePreset: key, objectives: { ...preset.objectives } });
    } else {
      set({ activePreset: "custom", objectives: { ...DEFAULT_OBJECTIVES } });
    }
  },

  setObjective: (target, config) => {
    set((s) => ({
      objectives: { ...s.objectives, [target]: config },
      activePreset: "custom",
    }));
  },

  setPopSize: (size) => set({ popSize: size }),
  setNGenerations: (n) => set({ nGenerations: n }),

  runOptimization: async () => {
    const { selectedModelIds, objectives, popSize, nGenerations, activePreset } =
      get();

    // Filter to only selected targets with models
    const activeModelIds: Record<string, string> = {};
    const activeObjectives: Record<string, ObjectiveConfig> = {};
    for (const [target, modelId] of Object.entries(selectedModelIds)) {
      if (modelId) {
        activeModelIds[target] = modelId;
        if (objectives[target]) {
          activeObjectives[target] = objectives[target];
        }
      }
    }

    if (Object.keys(activeModelIds).length === 0) {
      set({ optimizationError: "Select at least one model to optimize" });
      return;
    }

    set({
      optimizationLoading: true,
      optimizationError: null,
      solutions: [],
      convergence: [],
      optimizationStats: null,
    });

    try {
      const result = await runOptimization({
        model_ids: activeModelIds,
        objectives: activeObjectives,
        preset: activePreset,
        pop_size: popSize,
        n_generations: nGenerations,
      });

      if (result.error) {
        set({ optimizationLoading: false, optimizationError: result.error });
        return;
      }

      set({
        solutions: result.solutions,
        convergence: result.convergence,
        optimizationLoading: false,
        optimizationStats: {
          n_generations_run: result.n_generations_run,
          n_evaluations: result.n_evaluations,
          duration_seconds: result.duration_seconds,
          targets_optimized: result.targets_optimized,
        },
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Optimization failed";
      set({ optimizationLoading: false, optimizationError: msg });
    }
  },

  clearOptimization: () => {
    set({
      solutions: [],
      convergence: [],
      optimizationError: null,
      optimizationStats: null,
    });
  },
}));
