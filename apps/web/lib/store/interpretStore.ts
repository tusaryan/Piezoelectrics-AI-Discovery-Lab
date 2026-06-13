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
  fetchBeeswarmStatus,
  runShapWaterfall,
  runShapDependence,
  runPhysicsValidation,
  runSymbolicRegression,
  installPySRBackend as apiInstallPySRBackend,
} from "@/lib/api/interpret";

/** Classify interpretability errors into user-friendly messages. */
function _classifyInterpretError(e: unknown, context: string): string {
  if (!(e instanceof Error)) return `${context} failed unexpectedly.`;
  const raw = e.message;
  if (raw.includes("socket hang up") || raw.includes("ECONNRESET") || raw.includes("Failed to fetch")) {
    return `Server crashed during ${context}. The model may be incompatible or too large. Check terminal logs and try again.`;
  }
  if (raw.includes("TimeoutError") || raw.includes("AbortError")) {
    return `${context} timed out. The computation may need more time or the model is too complex.`;
  }
  if (raw.includes("Internal Server Error")) {
    return `Backend error during ${context}. Check terminal logs for details.`;
  }
  return raw;
}

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
  checkCachedBeeswarm: (modelId: string) => Promise<void>;
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
      beeswarmLoading: false,
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
    // Auto-check if beeswarm was previously computed (cached on server)
    get().checkCachedBeeswarm(id);
  },

  checkCachedBeeswarm: async (modelId: string) => {
    try {
      const status = await fetchBeeswarmStatus(modelId);
      if (status.status === "completed" && status.result) {
        // Only apply if user hasn't switched models while we were checking
        if (get().selectedModelId === modelId) {
          set({ beeswarm: status.result, beeswarmLoading: false });
          console.info("[Interpret] Loaded cached beeswarm for", modelId.slice(0, 8));
        }
      } else if (status.status === "computing") {
        // A background computation is running — show loading state and start polling
        if (get().selectedModelId === modelId) {
          set({ beeswarmLoading: true, beeswarmError: null });
          console.info("[Interpret] Beeswarm still computing for", modelId.slice(0, 8), "— will poll...");
          // Start polling
          get().fetchBeeswarm();
        }
      }
    } catch {
      // Silently ignore cache check failures
    }
  },

  fetchBeeswarm: async () => {
    const { selectedModelId } = get();
    if (!selectedModelId) return;
    set({ beeswarmLoading: true, beeswarmError: null });
    try {
      const result = await runShapBeeswarm(selectedModelId);
      // Only apply if user hasn't switched models while we were computing
      if (get().selectedModelId === selectedModelId) {
        set({ beeswarm: result, beeswarmLoading: false });
      }
    } catch (e: unknown) {
      if (get().selectedModelId !== selectedModelId) return; // stale
      const msg = _classifyInterpretError(e, "SHAP beeswarm analysis");
      console.error("[Interpret] Beeswarm failed:", msg, e);
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
      const msg = _classifyInterpretError(e, "SHAP waterfall analysis");
      console.error("[Interpret] Waterfall failed:", msg, e);
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
      const msg = _classifyInterpretError(e, `SHAP dependence for '${featureName}'`);
      console.error("[Interpret] Dependence failed:", msg, e);
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
      const msg = _classifyInterpretError(e, "physics validation");
      console.error("[Interpret] Physics validation failed:", msg, e);
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
      const msg = _classifyInterpretError(e, "symbolic regression");
      console.error("[Interpret] Symbolic regression failed:", msg, e);
      set({ symRegError: msg, symRegLoading: false });
    }
  },

  installPySRBackend: async () => {
    set({ pysrInstalling: true, symRegError: null });
    try {
      await apiInstallPySRBackend();
      set({ pysrInstalling: false });
      // Do NOT auto-run symbolic regression — let the user click "Run" manually
      // after installation completes in the background
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Installation failed";
      set({ symRegError: msg, pysrInstalling: false });
    }
  },

  reset: () => set(initialState),
}));
