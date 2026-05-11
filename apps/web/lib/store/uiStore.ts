import { create } from "zustand";

/**
 * UI state store — sidebar, layout preferences, and global feature flags.
 */

interface UIState {
  /** Whether the sidebar is collapsed to icon-only mode */
  sidebarCollapsed: boolean;

  /** Strict formula validation mode — catches invalid element patterns */
  strictFormulaMode: boolean;

  /** Navigation loading state */
  isNavigating: boolean;

  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setStrictFormulaMode: (strict: boolean) => void;
  setIsNavigating: (navigating: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarCollapsed: false,
  strictFormulaMode: true, // Default: strict mode ON for better safety
  isNavigating: false,

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  setStrictFormulaMode: (strict) => set({ strictFormulaMode: strict }),
  setIsNavigating: (navigating) => set({ isNavigating: navigating }),
}));
