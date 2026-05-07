import { create } from "zustand";

/**
 * UI state store — sidebar and layout preferences.
 */

interface UIState {
  /** Whether the sidebar is collapsed to icon-only mode */
  sidebarCollapsed: boolean;

  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarCollapsed: false,

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
}));
