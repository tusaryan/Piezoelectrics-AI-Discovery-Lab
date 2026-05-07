/**
 * Piezo.AI — Central App Constants
 *
 * Single source of truth for app branding, version, and developer info.
 * Change here → reflects everywhere (sidebar, header, footer, mobile).
 */

export const APP_CONFIG = {
  /** App display name */
  name: "Piezo.AI",

  /** Current version — later can be driven from env variable */
  version: "v2.1.0",

  /** Short logo text (used until an image logo is added) */
  logoText: "P",

  /** Full tagline */
  tagline: "AI-Driven Piezoelectric Material Discovery",

  /** Developer info */
  developer: {
    name: "Aryan Kumar",
    github: "https://github.com/tusaryan",
    linkedin: "https://linkedin.com/in/aryan-kumar-0b40292a9",
  },

  /** API endpoints (defaults — will be configurable in Settings) */
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    healthEndpoint: "/health",
  },
} as const;
