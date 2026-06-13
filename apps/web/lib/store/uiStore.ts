import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

/**
 * UI state store — sidebar, layout preferences, and global feature flags.
 * Uses zustand persist for state persistence across page reloads.
 */

interface UIState {
  /** Whether the sidebar is collapsed to icon-only mode */
  sidebarCollapsed: boolean;

  /** Strict formula validation mode — catches invalid element patterns */
  strictFormulaMode: boolean;

  /** Navigation loading state */
  isNavigating: boolean;

  /** Active settings tab */
  activeSettingsTab: string;

  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setStrictFormulaMode: (strict: boolean) => void;
  setIsNavigating: (navigating: boolean) => void;
  setActiveSettingsTab: (tab: string) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      strictFormulaMode: true, // Default: strict mode ON for better safety
      isNavigating: false,
      activeSettingsTab: "overview",

      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setStrictFormulaMode: (strict) => set({ strictFormulaMode: strict }),
      setIsNavigating: (navigating) => set({ isNavigating: navigating }),
      setActiveSettingsTab: (tab) => set({ activeSettingsTab: tab }),
    }),
    {
      name: "piezo-ui-store",
      storage: createJSONStorage(() => {
        // SSR-safe: only use localStorage on client
        if (typeof window !== "undefined") return localStorage;
        return {
          getItem: () => null,
          setItem: () => {},
          removeItem: () => {},
        };
      }),
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        strictFormulaMode: state.strictFormulaMode,
        activeSettingsTab: state.activeSettingsTab,
      }),
    }
  )
);
