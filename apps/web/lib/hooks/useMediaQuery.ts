"use client";

import { useState, useEffect, useCallback } from "react";

/**
 * Custom hook for responsive breakpoint detection.
 * SSR-safe — returns false on server, updates on client via matchMedia.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);

    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [query]);

  return matches;
}

/* ---------- Named Breakpoint Hooks ---------- */

/** XL: ≥1440px — Full layout, expanded sidebar */
export function useIsXL(): boolean {
  return useMediaQuery("(min-width: 1440px)");
}

/** LG: 1080–1439px — Icon-only sidebar by default */
export function useIsLG(): boolean {
  return useMediaQuery("(min-width: 1080px) and (max-width: 1439px)");
}

/** MD: 768–1079px — Single-column, icon-only sidebar */
export function useIsMD(): boolean {
  return useMediaQuery("(min-width: 768px) and (max-width: 1079px)");
}

/** SM: <768px — Mobile: hamburger + bottom nav */
export function useIsSM(): boolean {
  return useMediaQuery("(max-width: 767px)");
}

/** Desktop: ≥768px — Sidebar visible */
export function useIsDesktop(): boolean {
  return useMediaQuery("(min-width: 768px)");
}

/** Wide desktop: ≥1440px — Expanded sidebar */
export function useIsWideDesktop(): boolean {
  return useMediaQuery("(min-width: 1440px)");
}
