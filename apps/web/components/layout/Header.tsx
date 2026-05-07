"use client";

import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useState, useCallback } from "react";
import {
  Moon,
  Sun,
  Lamp,
  Github,
  Linkedin,
} from "lucide-react";
import { useIsSM } from "@/lib/hooks/useMediaQuery";
import { APP_CONFIG } from "@/lib/constants";

/**
 * Route → page title mapping for the header.
 */
const PAGE_TITLES: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/dataset": "Dataset",
  "/train": "Model Studio",
  "/predict": "Predict",
  "/optimization-lab": "Optimization Lab",
  "/interpret": "Interpretability",
  "/settings": "Settings",
};

/** Theme cycle order and their display icons */
const THEME_CYCLE = ["dark", "light", "night"] as const;

const THEME_ICONS: Record<string, typeof Moon> = {
  dark: Moon,
  light: Sun,
  night: Lamp,
};

const THEME_LABELS: Record<string, string> = {
  dark: "Dark",
  light: "Light",
  night: "Night",
};

export default function Header() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const isSM = useIsSM();
  const [mounted, setMounted] = useState(false);
  const [isOnline, setIsOnline] = useState(false);

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // System status check — polls /health endpoint
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(APP_CONFIG.api.healthEndpoint, {
        signal: AbortSignal.timeout(3000),
      });
      setIsOnline(res.ok);
    } catch {
      setIsOnline(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Cycle theme: dark → light → night → dark
  const cycleTheme = () => {
    const current = theme || "dark";
    const currentIndex = THEME_CYCLE.indexOf(current as (typeof THEME_CYCLE)[number]);
    const nextIndex = (currentIndex + 1) % THEME_CYCLE.length;
    setTheme(THEME_CYCLE[nextIndex]);
  };

  // Get page title from route
  const pageTitle = PAGE_TITLES[pathname] || APP_CONFIG.name;

  // Current theme info
  const currentTheme = (mounted ? theme : "dark") || "dark";
  const ThemeIcon = THEME_ICONS[currentTheme] || Moon;
  const themeLabel = THEME_LABELS[currentTheme] || "Dark";

  return (
    <header className="app-header">
      <div className="header-left">
        {/* Piezo.AI logo — shown on mobile since sidebar is hidden */}
        {isSM && (
          <div className="header-brand">
            <div className="header-brand-logo">{APP_CONFIG.logoText}</div>
          </div>
        )}
        <h2 className="header-page-title">{pageTitle}</h2>
      </div>

      <div className="header-right">
        {/* Developer links + version — shown on mobile since sidebar is hidden */}
        {isSM && (
          <div className="header-meta">
            <a
              href={APP_CONFIG.developer.github}
              target="_blank"
              rel="noopener noreferrer"
              className="header-meta-link"
              title="Developer GitHub"
              aria-label="Developer GitHub"
            >
              <Github size={15} />
            </a>
            <a
              href={APP_CONFIG.developer.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="header-meta-link"
              title="Developer LinkedIn"
              aria-label="Developer LinkedIn"
            >
              <Linkedin size={15} />
            </a>
            <span className="header-meta-version">{APP_CONFIG.version}</span>
          </div>
        )}

        {/* System status indicator */}
        <div className="status-indicator" title={isOnline ? "System online" : "System offline"}>
          <span className={`status-dot ${isOnline ? "online" : "offline"}`} />
          <span className="status-label">{isOnline ? "Online" : "Offline"}</span>
        </div>

        {/* Theme toggle */}
        {mounted && (
          <button
            className="header-btn"
            onClick={cycleTheme}
            aria-label={`Current theme: ${themeLabel}. Click to switch.`}
            title={`Theme: ${themeLabel}`}
          >
            <ThemeIcon size={18} />
          </button>
        )}
      </div>
    </header>
  );
}
