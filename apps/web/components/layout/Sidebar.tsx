"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart3,
  Database,
  BrainCircuit,
  Zap,
  FlaskConical,
  Eye,
  Settings,
  ChevronLeft,
  ChevronRight,
  Github,
  Linkedin,
} from "lucide-react";
import { useUIStore } from "@/lib/store/uiStore";
import { useIsWideDesktop, useIsSM } from "@/lib/hooks/useMediaQuery";
import { useAppBranding } from "@/lib/hooks/useAppBranding";
import { useEffect } from "react";

/**
 * Navigation items — maps to the 7 sections defined in §2 of architecture spec.
 */
const NAV_ITEMS = [
  { label: "Dashboard", route: "/dashboard", icon: BarChart3 },
  { label: "Dataset", route: "/dataset", icon: Database },
  { label: "Train", route: "/train", icon: BrainCircuit },
  { label: "Predict", route: "/predict", icon: Zap },
  { label: "Optimization Lab", route: "/optimization-lab", icon: FlaskConical },
  { label: "Interpretability", route: "/interpret", icon: Eye },
  { label: "Settings", route: "/settings", icon: Settings },
] as const;

export default function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar, setSidebarCollapsed } = useUIStore();
  const isWideDesktop = useIsWideDesktop();
  const isSM = useIsSM();
  const branding = useAppBranding();

  // Auto-collapse sidebar on narrower screens
  useEffect(() => {
    if (!isWideDesktop && !isSM) {
      setSidebarCollapsed(true);
    } else if (isWideDesktop) {
      setSidebarCollapsed(false);
    }
  }, [isWideDesktop, isSM, setSidebarCollapsed]);

  const isActive = (route: string) => {
    if (route === "/dashboard") return pathname === "/" || pathname === "/dashboard";
    return pathname.startsWith(route);
  };

  const sidebarClasses = [
    "app-sidebar",
    sidebarCollapsed ? "collapsed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <motion.aside
      className={sidebarClasses}
      initial={false}
      animate={{
        width: sidebarCollapsed ? 72 : 260,
      }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {/* Brand — pinned top */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-logo" aria-label={branding.name} suppressHydrationWarning>
          <img
            src={branding.logoPath}
            alt={branding.name}
            suppressHydrationWarning
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
              const parent = (e.target as HTMLElement).parentElement;
              if (parent && !parent.querySelector(".sidebar-logo-fallback")) {
                const span = document.createElement("span");
                span.className = "sidebar-logo-fallback";
                span.textContent = branding.logoText;
                parent.appendChild(span);
              }
            }}
          />
        </div>
        <AnimatePresence mode="wait">
          {!sidebarCollapsed && (
            <motion.div
              className="sidebar-brand-text"
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: "auto" }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.2 }}
              suppressHydrationWarning
            >
              <span className="sidebar-brand-name" suppressHydrationWarning>{branding.name}</span>
              <span className="sidebar-brand-version" suppressHydrationWarning>{branding.version}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation — scrollable area */}
      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.route);

          return (
            <Link
              key={item.route}
              href={item.route}
              className={`sidebar-nav-item ${active ? "active" : ""}`}
              title={sidebarCollapsed ? item.label : undefined}
              aria-current={active ? "page" : undefined}
            >
              <Icon className="nav-icon" size={20} strokeWidth={active ? 2.2 : 1.8} />
              <span className="nav-label">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer — pinned bottom: version + social links */}
      <div className="sidebar-footer">
        <div className={`sidebar-footer-content ${sidebarCollapsed ? "collapsed-footer" : ""}`}>
          <div className={`sidebar-footer-links ${sidebarCollapsed ? "stacked" : ""}`}>
            <a
              href={branding.developer.github}
              target="_blank"
              rel="noopener noreferrer"
              className="sidebar-footer-link"
              title="Developer GitHub"
              aria-label="Developer GitHub"
            >
              <Github size={16} />
            </a>
            <a
              href={branding.developer.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="sidebar-footer-link"
              title="Developer LinkedIn"
              aria-label="Developer LinkedIn"
            >
              <Linkedin size={16} />
            </a>
          </div>
          <span className="sidebar-footer-version" suppressHydrationWarning>{branding.version}</span>
        </div>
      </div>

      {/* Collapse toggle — pinned bottom */}
      <button
        className="sidebar-toggle"
        onClick={toggleSidebar}
        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {sidebarCollapsed ? (
          <ChevronRight size={16} />
        ) : (
          <>
            <ChevronLeft size={16} />
            <span className="toggle-label">Collapse</span>
          </>
        )}
      </button>
    </motion.aside>
  );
}
