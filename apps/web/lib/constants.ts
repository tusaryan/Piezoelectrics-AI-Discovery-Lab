/**
 * Piezo.AI — Central App Constants
 *
 * Single source of truth for app branding, version, and developer info.
 * All values are driven by NEXT_PUBLIC_* env variables (set in .env).
 *
 * To change branding (name, version, logo):
 *   1. Edit .env → restart dev server
 *   2. Everything updates automatically (sidebar, header, footer, mobile)
 */

export const APP_CONFIG = {
  /** App display name */
  name: process.env.NEXT_PUBLIC_APP_NAME || "Piezo.AI",

  /** Current version — driven by NEXT_PUBLIC_APP_VERSION */
  version: `v${process.env.NEXT_PUBLIC_APP_VERSION || "2.1.0"}`,

  /** Short logo text (used until an image logo is added) */
  logoText: process.env.NEXT_PUBLIC_APP_LOGO_TEXT || "P",

  /** Full tagline */
  tagline: "AI-Driven Piezoelectric Material Discovery",

  /** Developer info — driven by env variables */
  developer: {
    name: process.env.NEXT_PUBLIC_DEV_NAME || "Aryan",
    github: process.env.NEXT_PUBLIC_DEV_GITHUB || "https://github.com/tusaryan",
    linkedin: process.env.NEXT_PUBLIC_DEV_LINKEDIN || "https://www.linkedin.com/in/tusaryan/",
  },

  /** API endpoints (defaults — configurable via Settings in S9) */
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    wsUrl: process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000",
    healthEndpoint: "/health",
  },
} as const;
