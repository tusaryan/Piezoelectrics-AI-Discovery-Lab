/**
 * Interpret Store — Zustand state management for Interpretability section.
 */

import { create } from "zustand";
import type {
  InterpretModel,
  ShapBeeswarmResult,
  ShapWaterfallResult,
  ShapDependenceResult,
  PhysicsValidationResult,
  SymbolicRegressionResult,
} from "@/lib/api/interpret";
import {
  fetchInterpretModels,
  runShapBeeswarm,
  runShapWaterfall,
  runShapDependence,
  runPhysicsValidation,
  runSymbolicRegression,
  installPySRBackend as apiInstallPySRBackend,
} from "@/lib/api/interpret";

interface InterpretState {
  // Models
  models: InterpretModel[];
  selectedModelId: string | null;
  modelsLoading: boolean;

  // SHAP Beeswarm
  beeswarm: ShapBeeswarmResult | null;
  beeswarmLoading: boolean;
  beeswarmError: string | null;

  // SHAP Waterfall
  waterfall: ShapWaterfallResult | null;
  waterfallLoading: boolean;
  waterfallError: string | null;
  waterfallSampleIndex: number;

  // SHAP Dependence
  dependence: ShapDependenceResult | null;
  dependenceLoading: boolean;
  dependenceError: string | null;
  dependenceFeature: string | null;

  // Physics Validation
  physics: PhysicsValidationResult | null;
  physicsLoading: boolean;
  physicsError: string | null;

  // Symbolic Regression
  symbolicRegression: SymbolicRegressionResult | null;
  symRegLoading: boolean;
  symRegError: string | null;

  // Actions
  loadModels: () => Promise<void>;
  selectModel: (id: string) => void;
  fetchBeeswarm: () => Promise<void>;
  fetchWaterfall: (sampleIndex?: number) => Promise<void>;
  fetchDependence: (featureName: string) => Promise<void>;
  fetchPhysicsValidation: () => Promise<void>;
  fetchSymbolicRegression: (opts?: {
    maxComplexity?: number;
    nIterations?: number;
    timeoutSeconds?: number;
  }) => Promise<void>;
  installPySRBackend: () => Promise<void>;
  pysrInstalling: boolean;
  reset: () => void;
}

const initialState = {
  models: [],
  selectedModelId: null,
  modelsLoading: false,
  beeswarm: null,
  beeswarmLoading: false,
  beeswarmError: null,
  waterfall: null,
  waterfallLoading: false,
  waterfallError: null,
  waterfallSampleIndex: 0,
  dependence: null,
  dependenceLoading: false,
  dependenceError: null,
  dependenceFeature: null,
  physics: null,
  physicsLoading: false,
  physicsError: null,
  symbolicRegression: null,
  symRegLoading: false,
  symRegError: null,
  pysrInstalling: false,
};

export const useInterpretStore = create<InterpretState>((set, get) => ({
  ...initialState,

  loadModels: async () => {
    set({ modelsLoading: true });
    try {
      const models = await fetchInterpretModels();
      set({ models, modelsLoading: false });
    } catch {
      set({ modelsLoading: false });
    }
  },

  selectModel: (id: string) => {
    set({
      selectedModelId: id,
      beeswarm: null,
      beeswarmError: null,
      waterfall: null,
      waterfallError: null,
      dependence: null,
      dependenceError: null,
      physics: null,
      physicsError: null,
      symbolicRegression: null,
      symRegError: null,
      waterfallSampleIndex: 0,
      dependenceFeature: null,
    });
  },

  fetchBeeswarm: async () => {
    const { selectedModelId } = get();
    if (!selectedModelId) return;
    set({ beeswarmLoading: true, beeswarmError: null });
    try {
      const result = await runShapBeeswarm(selectedModelId);
      set({ beeswarm: result, beeswarmLoading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "SHAP analysis failed";
      set({ beeswarmError: msg, beeswarmLoading: false });
    }
  },

  fetchWaterfall: async (sampleIndex?: number) => {
    const { selectedModelId } = get();
    if (!selectedModelId) return;
    const idx = sampleIndex ?? get().waterfallSampleIndex;
    set({ waterfallLoading: true, waterfallError: null, waterfallSampleIndex: idx });
    try {
      const result = await runShapWaterfall(selectedModelId, idx);
      set({ waterfall: result, waterfallLoading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Waterfall failed";
      set({ waterfallError: msg, waterfallLoading: false });
    }
  },

  fetchDependence: async (featureName: string) => {
    const { selectedModelId } = get();
    if (!selectedModelId) return;
    set({ dependenceLoading: true, dependenceError: null, dependenceFeature: featureName });
    try {
      const result = await runShapDependence(selectedModelId, featureName);
      set({ dependence: result, dependenceLoading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Dependence failed";
      set({ dependenceError: msg, dependenceLoading: false });
    }
  },

  fetchPhysicsValidation: async () => {
    const { selectedModelId } = get();
    if (!selectedModelId) return;
    set({ physicsLoading: true, physicsError: null });
    try {
      const result = await runPhysicsValidation(selectedModelId);
      set({ physics: result, physicsLoading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Physics validation failed";
      set({ physicsError: msg, physicsLoading: false });
    }
  },

  fetchSymbolicRegression: async (opts) => {
    const { selectedModelId } = get();
    if (!selectedModelId) return;
    set({ symRegLoading: true, symRegError: null });
    try {
      const result = await runSymbolicRegression(selectedModelId, opts);
      set({ symbolicRegression: result, symRegLoading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Symbolic regression failed";
      set({ symRegError: msg, symRegLoading: false });
    }
  },

  installPySRBackend: async () => {
    set({ pysrInstalling: true, symRegError: null });
    try {
      await apiInstallPySRBackend();
      set({ pysrInstalling: false });
      // Clear the error/availability state by triggering a rerun of symReg
      // Or just let the user click "Run" again.
      // We will re-fetch the symbolic regression to clear the 'not available' state
      get().fetchSymbolicRegression();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Installation failed";
      set({ symRegError: msg, pysrInstalling: false });
    }
  },

  reset: () => set(initialState),
}));
