"use client";

import { type ReactNode } from "react";
import Sidebar from "./Sidebar";
import Header from "./Header";
import BottomNav from "./BottomNav";
import { useUIStore } from "@/lib/store/uiStore";

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

  return (
    <div className="app-shell" data-sidebar-collapsed={sidebarCollapsed}>
      <Sidebar />
      <Header />
      <main className="app-main">
        {children}
      </main>
      <BottomNav />
    </div>
  );
}
