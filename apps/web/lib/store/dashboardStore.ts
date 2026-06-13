/**
 * Piezo.AI — Dashboard Store (Zustand)
 * ========================================
 * State management for the dashboard page.
 */

import { create } from "zustand";
import type {
  SystemStats,
  DashboardModel,
  TargetDistribution,
  PredictionHistoryItem,
} from "@/lib/api/dashboard";
import {
  getSystemStats,
  listDashboardModels,
  getTargetDistribution,
  getPredictionHistory,
  renameDashboardModel,
  setDashboardModelDefault,
  deleteDashboardModel,
  bulkDeleteDashboardModels,
  deletePrediction as apiDeletePrediction,
  bulkDeletePredictions as apiBulkDeletePredictions,
  generateReport,
  type ReportGenerateRequest,
  type ReportGenerateResponse,
} from "@/lib/api/dashboard";

interface DashboardState {
  // Data
  stats: SystemStats | null;
  models: DashboardModel[];
  targetDistribution: TargetDistribution[];
  predictionHistory: PredictionHistoryItem[];

  // UI state
  loading: boolean;
  error: string | null;
  reportGenerating: boolean;
  lastReport: ReportGenerateResponse | null;

  // Selected models for bulk operations
  selectedModelIds: Set<string>;

  // Actions
  fetchAll: () => Promise<void>;
  fetchStats: () => Promise<void>;
  fetchModels: () => Promise<void>;
  fetchTargetDistribution: () => Promise<void>;
  fetchPredictionHistory: () => Promise<void>;
  renameModel: (id: string, name: string) => Promise<void>;
  setDefaultModel: (id: string) => Promise<void>;
  deleteModel: (id: string) => Promise<void>;
  bulkDeleteModels: () => Promise<void>;
  toggleModelSelection: (id: string) => void;
  selectAllModels: () => void;
  clearModelSelection: () => void;
  generateReport: (options: ReportGenerateRequest) => Promise<void>;
  deletePrediction: (id: string) => Promise<void>;
  bulkDeletePredictions: (ids: string[]) => Promise<void>;
  clearError: () => void;
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  stats: null,
  models: [],
  targetDistribution: [],
  predictionHistory: [],
  loading: false,
  error: null,
  reportGenerating: false,
  lastReport: null,
  selectedModelIds: new Set(),

  fetchAll: async () => {
    set({ loading: true, error: null });
    try {
      const [stats, models, dist, history] = await Promise.all([
        getSystemStats(),
        listDashboardModels(),
        getTargetDistribution(),
        getPredictionHistory(100),
      ]);
      set({ stats, models, targetDistribution: dist, predictionHistory: history, loading: false });
    } catch (e: unknown) {
      set({ loading: false, error: e instanceof Error ? e.message : "Failed to load dashboard" });
    }
  },

  fetchStats: async () => {
    try {
      const stats = await getSystemStats();
      set({ stats });
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to load stats" });
    }
  },

  fetchModels: async () => {
    try {
      const models = await listDashboardModels();
      set({ models });
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to load models" });
    }
  },

  fetchTargetDistribution: async () => {
    try {
      const dist = await getTargetDistribution();
      set({ targetDistribution: dist });
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to load distribution" });
    }
  },

  fetchPredictionHistory: async () => {
    try {
      const history = await getPredictionHistory(100);
      set({ predictionHistory: history });
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to load history" });
    }
  },

  renameModel: async (id: string, name: string) => {
    try {
      const updated = await renameDashboardModel(id, name);
      set((s) => ({
        models: s.models.map((m) => (m.id === id ? { ...m, display_name: updated.display_name } : m)),
      }));
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to rename model" });
    }
  },

  setDefaultModel: async (id: string) => {
    try {
      const updated = await setDashboardModelDefault(id);
      set((s) => ({
        models: s.models.map((m) => {
          if (m.target === updated.target) {
            return { ...m, is_default: m.id === id };
          }
          return m;
        }),
      }));
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to set default" });
    }
  },

  deleteModel: async (id: string) => {
    try {
      await deleteDashboardModel(id);
      set((s) => ({
        models: s.models.filter((m) => m.id !== id),
        selectedModelIds: (() => {
          const next = new Set(s.selectedModelIds);
          next.delete(id);
          return next;
        })(),
      }));
      // Refresh stats after deletion
      get().fetchStats();
      get().fetchTargetDistribution();
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to delete model" });
    }
  },

  bulkDeleteModels: async () => {
    const ids = Array.from(get().selectedModelIds);
    if (ids.length === 0) return;
    try {
      await bulkDeleteDashboardModels(ids);
      set((s) => ({
        models: s.models.filter((m) => !ids.includes(m.id)),
        selectedModelIds: new Set(),
      }));
      get().fetchStats();
      get().fetchTargetDistribution();
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to delete models" });
    }
  },

  toggleModelSelection: (id: string) => {
    set((s) => {
      const next = new Set(s.selectedModelIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { selectedModelIds: next };
    });
  },

  selectAllModels: () => {
    set((s) => ({
      selectedModelIds: new Set(s.models.map((m) => m.id)),
    }));
  },

  clearModelSelection: () => {
    set({ selectedModelIds: new Set() });
  },

  generateReport: async (options: ReportGenerateRequest) => {
    set({ reportGenerating: true, error: null });
    try {
      const result = await generateReport(options);
      set({ reportGenerating: false, lastReport: result });
    } catch (e: unknown) {
      set({
        reportGenerating: false,
        error: e instanceof Error ? e.message : "Report generation failed",
      });
    }
  },

  deletePrediction: async (id: string) => {
    try {
      await apiDeletePrediction(id);
      set((s) => ({
        predictionHistory: s.predictionHistory.filter((p) => p.id !== id),
      }));
      get().fetchStats();
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to delete prediction" });
    }
  },

  bulkDeletePredictions: async (ids: string[]) => {
    if (ids.length === 0) return;
    try {
      await apiBulkDeletePredictions(ids);
      set((s) => ({
        predictionHistory: s.predictionHistory.filter((p) => !ids.includes(p.id)),
      }));
      get().fetchStats();
    } catch (e: unknown) {
      set({ error: e instanceof Error ? e.message : "Failed to delete predictions" });
    }
  },

  clearError: () => set({ error: null }),
}));
