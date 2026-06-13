"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import { type ReactNode } from "react";

/**
 * ThemeProvider — wraps the app with next-themes for dark/light/night switching.
 *
 * Uses `data-theme` attribute on <html> to drive CSS variable switching.
 * 3 themes: dark (default), light, night (warm amber for eye protection).
 */

interface ThemeProviderProps {
  children: ReactNode;
}

export default function ThemeProvider({ children }: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="data-theme"
      defaultTheme="dark"
      themes={["dark", "light", "night"]}
      storageKey="piezo-theme"
      disableTransitionOnChange={false}
    >
      {children}
    </NextThemesProvider>
  );
}
