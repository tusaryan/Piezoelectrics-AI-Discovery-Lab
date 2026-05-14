/**
 * useAppBranding — Central, reactive hook for app branding.
 *
 * Priority order (highest → lowest):
 *   1. Runtime API config (settingsStore.appConfig) — updated via Settings UI / logo upload
 *   2. NEXT_PUBLIC_* env vars (build-time from .env)
 *   3. Hardcoded defaults
 *
 * This ensures:
 *   - Uploaded logos take effect immediately (no reload needed)
 *   - .env changes work after rebuild
 *   - Defaults always provide a fallback
 */

import { useMemo, useEffect } from "react";
import { useSettingsStore } from "@/lib/store/settingsStore";

const DEFAULTS = {
  name: "Piezo.AI",
  version: "2.1.0",
  logoText: "P",
  logoPath: "/piezo-ai-logo.png",
  tagline: "AI-Driven Piezoelectric Material Discovery",
  devName: "Aryan",
  devGithub: "https://github.com/tusaryan",
  devLinkedin: "https://www.linkedin.com/in/tusaryan/",
} as const;

export function useAppBranding() {
  const appConfig = useSettingsStore((s) => s.appConfig);

  const branding = useMemo(() => {
    // Helper: pick first non-empty value from candidates
    const pick = (...candidates: (string | undefined | null)[]) =>
      candidates.find((v) => v != null && v.trim() !== "") ?? "";

    const name = pick(
      appConfig.APP_NAME,
      process.env.NEXT_PUBLIC_APP_NAME,
      DEFAULTS.name,
    );

    const rawVersion = pick(
      appConfig.APP_VERSION,
      process.env.NEXT_PUBLIC_APP_VERSION,
      DEFAULTS.version,
    );
    const version = rawVersion.startsWith("v") ? rawVersion : `v${rawVersion}`;

    const logoText = pick(
      appConfig.APP_LOGO_TEXT,
      process.env.NEXT_PUBLIC_APP_LOGO_TEXT,
      DEFAULTS.logoText,
    );

    const logoPath = pick(
      appConfig.APP_LOGO_PATH,
      appConfig.NEXT_PUBLIC_APP_LOGO_PATH,
      process.env.NEXT_PUBLIC_APP_LOGO_PATH,
      DEFAULTS.logoPath,
    );

    const tagline = pick(
      appConfig.APP_TAGLINE,
      DEFAULTS.tagline,
    );

    const devName = pick(
      appConfig.NEXT_PUBLIC_DEV_NAME,
      process.env.NEXT_PUBLIC_DEV_NAME,
      DEFAULTS.devName,
    );

    const devGithub = pick(
      appConfig.NEXT_PUBLIC_DEV_GITHUB,
      process.env.NEXT_PUBLIC_DEV_GITHUB,
      DEFAULTS.devGithub,
    );

    const devLinkedin = pick(
      appConfig.NEXT_PUBLIC_DEV_LINKEDIN,
      process.env.NEXT_PUBLIC_DEV_LINKEDIN,
      DEFAULTS.devLinkedin,
    );

    return {
      name,
      version,
      logoText,
      logoPath,
      tagline,
      developer: {
        name: devName,
        github: devGithub,
        linkedin: devLinkedin,
      },
    };
  }, [appConfig]);

  // Dynamically update the favicon (Chrome tab logo) when the branding logo changes
  useEffect(() => {
    if (typeof document === "undefined") return;

    const links = document.querySelectorAll("link[rel~='icon']");
    if (links.length > 0) {
      links.forEach((link) => {
        (link as HTMLLinkElement).href = branding.logoPath;
      });
    } else {
      const link = document.createElement("link");
      link.rel = "icon";
      link.href = branding.logoPath;
      document.head.appendChild(link);
    }
  }, [branding.logoPath]);

  return branding;
}
