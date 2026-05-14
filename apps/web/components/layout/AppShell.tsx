"use client";

import { type ReactNode, useEffect } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";
import BottomNav from "./BottomNav";
import NavigationLoader from "@/components/common/NavigationLoader";
import { useUIStore } from "@/lib/store/uiStore";
import { useSettingsStore } from "@/lib/store/settingsStore";

/**
 * AppShell — Main layout composition.
 *
 * Sidebar is position:fixed, so header/main use margin-left.
 * The `data-sidebar-collapsed` attribute drives the margin CSS.
 * Mobile (<768px): sidebar hidden via CSS, margins zeroed.
 */

interface AppShellProps {
  children: ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
  const { sidebarCollapsed } = useUIStore();
  const fetchAppConfig = useSettingsStore((s) => s.fetchAppConfig);

  // Fetch app config on mount so branding (logo, name, etc.) is available
  useEffect(() => {
    fetchAppConfig();
  }, [fetchAppConfig]);

  return (
    <div className="app-shell" data-sidebar-collapsed={sidebarCollapsed}>
      <NavigationLoader />
      <Sidebar />
      <Header />
      <main className="app-main">
        {children}
      </main>
      <BottomNav />
    </div>
  );
}
