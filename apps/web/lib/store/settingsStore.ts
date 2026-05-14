/**
 * Piezo.AI — Settings Store (Zustand)
 * =====================================
 * State management for all settings sub-sections.
 */

import { create } from "zustand";
import * as api from "../api/settings";

interface SettingsState {
  // System environment
  systemEnv: api.SystemEnvironment | null;
  systemLoading: boolean;

  // App config
  appConfig: api.AppConfig;
  configLoading: boolean;
  configSaving: boolean;

  // LLM
  llmConfig: api.LlmConfig | null;
  llmProviders: api.LlmProvider[];
  llmLoading: boolean;
  llmSaving: boolean;

  // Elements
  elementRegistry: api.ElementRegistry | null;
  elementsLoading: boolean;

  // Models
  models: api.SettingsModel[];
  modelsLoading: boolean;

  // GNN
  gnnStatus: api.GnnStatus | null;
  gnnLoading: boolean;

  // Error
  error: string | null;

  // Actions — System
  fetchSystemEnv: () => Promise<void>;

  // Actions — App Config
  fetchAppConfig: () => Promise<void>;
  saveAppConfig: (updates: Record<string, string>) => Promise<void>;
  importEnvFile: (file: File) => Promise<api.EnvImportResult>;

  // Actions — LLM
  fetchLlmConfig: () => Promise<void>;
  fetchLlmProviders: () => Promise<void>;
  saveLlmConfig: (data: Parameters<typeof api.updateLlmConfig>[0]) => Promise<void>;

  // Actions — Elements
  fetchElements: () => Promise<void>;
  addPendingElement: (symbol: string, categories: string[]) => Promise<void>;
  removePendingElement: (symbol: string) => Promise<void>;
  removeSupportedElement: (symbol: string) => Promise<{ message: string }>;
  bootstrapElements: () => Promise<string>;
  addCustomProperty: (key: string) => Promise<{ message: string }>;
  removeCustomProperty: (key: string) => Promise<{ message: string }>;
  resetElementsAndProperties: () => Promise<api.ResetResult>;

  // Actions — Models
  fetchModels: () => Promise<void>;
  renameModel: (id: string, name: string) => Promise<void>;
  setDefaultModel: (id: string) => Promise<void>;
  deleteModel: (id: string) => Promise<void>;
  batchDeleteModels: (ids: string[]) => Promise<api.DangerResult>;

  // Actions — Danger Zone
  purgeAllModels: () => Promise<api.DangerResult>;
  clearCache: () => Promise<api.DangerResult>;

  // Actions — Reset
  resetAllSettings: () => Promise<api.ResetResult>;

  // Actions — GNN
  fetchGnnStatus: () => Promise<void>;
  fetchAll: () => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  systemEnv: null,
  systemLoading: false,
  appConfig: typeof window !== "undefined" && localStorage.getItem("piezo_app_config") 
    ? JSON.parse(localStorage.getItem("piezo_app_config")!) 
    : {},
  configLoading: false,
  configSaving: false,
  llmConfig: null,
  llmProviders: [],
  llmLoading: false,
  llmSaving: false,
  elementRegistry: null,
  elementsLoading: false,
  models: [],
  modelsLoading: false,
  gnnStatus: null,
  gnnLoading: false,
  error: null,

  fetchSystemEnv: async () => {
    set({ systemLoading: true, error: null });
    try {
      const data = await api.getSystemEnvironment();
      set({ systemEnv: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ systemLoading: false }); }
  },

  fetchAppConfig: async () => {
    set({ configLoading: true, error: null });
    try {
      const data = await api.getAppConfig();
      if (typeof window !== "undefined") {
        localStorage.setItem("piezo_app_config", JSON.stringify(data));
      }
      set({ appConfig: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ configLoading: false }); }
  },

  saveAppConfig: async (updates) => {
    set({ configSaving: true, error: null });
    try {
      const data = await api.updateAppConfig(updates);
      if (typeof window !== "undefined") {
        localStorage.setItem("piezo_app_config", JSON.stringify(data));
      }
      set({ appConfig: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ configSaving: false }); }
  },

  importEnvFile: async (file) => {
    set({ configSaving: true, error: null });
    try {
      const result = await api.importEnvFile(file);
      await get().fetchAppConfig();
      return result;
    } catch (e: any) {
      set({ error: e.message });
      throw e;
    } finally { set({ configSaving: false }); }
  },

  fetchLlmConfig: async () => {
    set({ llmLoading: true, error: null });
    try {
      const data = await api.getLlmStatus();
      set({ llmConfig: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ llmLoading: false }); }
  },

  fetchLlmProviders: async () => {
    try {
      const data = await api.getLlmProviders();
      set({ llmProviders: data });
    } catch (e: any) { set({ error: e.message }); }
  },

  saveLlmConfig: async (data) => {
    set({ llmSaving: true, error: null });
    try {
      const updated = await api.updateLlmConfig(data);
      set({ llmConfig: updated });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ llmSaving: false }); }
  },

  fetchElements: async () => {
    set({ elementsLoading: true, error: null });
    try {
      const data = await api.getElementRegistry();
      set({ elementRegistry: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ elementsLoading: false }); }
  },

  addPendingElement: async (symbol, categories) => {
    try {
      await api.addPendingElement(symbol, categories);
      await get().fetchElements();
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  removePendingElement: async (symbol) => {
    try {
      await api.removePendingElement(symbol);
      await get().fetchElements();
    } catch (e: any) { set({ error: e.message }); }
  },

  removeSupportedElement: async (symbol) => {
    try {
      const result = await api.removeSupportedElement(symbol);
      await get().fetchElements();
      return result;
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  bootstrapElements: async () => {
    try {
      const result = await api.bootstrapElements();
      await get().fetchElements();
      return result.message;
    } catch (e: any) { set({ error: e.message }); return e.message; }
  },

  addCustomProperty: async (key) => {
    try {
      const result = await api.addCustomProperty(key);
      await get().fetchElements();
      return result;
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  removeCustomProperty: async (key) => {
    try {
      const result = await api.removeCustomProperty(key);
      await get().fetchElements();
      return result;
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  resetElementsAndProperties: async () => {
    try {
      const result = await api.resetElementsAndProperties();
      await get().fetchElements();
      return result;
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  fetchModels: async () => {
    set({ modelsLoading: true, error: null });
    try {
      const data = await api.getSettingsModels();
      set({ models: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ modelsLoading: false }); }
  },

  renameModel: async (id, name) => {
    try {
      await api.renameSettingsModel(id, name);
      await get().fetchModels();
    } catch (e: any) { set({ error: e.message }); }
  },

  setDefaultModel: async (id) => {
    try {
      await api.setDefaultSettingsModel(id);
      await get().fetchModels();
    } catch (e: any) { set({ error: e.message }); }
  },

  deleteModel: async (id) => {
    try {
      await api.deleteSettingsModel(id);
      await get().fetchModels();
    } catch (e: any) { set({ error: e.message }); }
  },

  batchDeleteModels: async (ids) => {
    try {
      const result = await api.batchDeleteModels(ids);
      await get().fetchModels();
      await get().fetchSystemEnv();
      return result;
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  purgeAllModels: async () => {
    const result = await api.purgeAllModels();
    await get().fetchModels();
    await get().fetchSystemEnv();
    return result;
  },

  clearCache: async () => {
    const result = await api.clearPredictionCache();
    await get().fetchSystemEnv();
    return result;
  },

  resetAllSettings: async () => {
    try {
      const result = await api.resetAllSettings();
      await get().fetchAll();
      return result;
    } catch (e: any) { set({ error: e.message }); throw e; }
  },

  fetchGnnStatus: async () => {
    set({ gnnLoading: true });
    try {
      const data = await api.getGnnStatus();
      set({ gnnStatus: data });
    } catch (e: any) { set({ error: e.message }); }
    finally { set({ gnnLoading: false }); }
  },

  fetchAll: async () => {
    const s = get();
    await Promise.allSettled([
      s.fetchSystemEnv(),
      s.fetchAppConfig(),
      s.fetchLlmConfig(),
      s.fetchLlmProviders(),
      s.fetchElements(),
      s.fetchModels(),
      s.fetchGnnStatus(),
    ]);
  },
}));
