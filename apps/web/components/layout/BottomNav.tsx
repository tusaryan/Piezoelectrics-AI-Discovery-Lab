"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Database,
  BrainCircuit,
  Zap,
  FlaskConical,
  Eye,
  Settings,
} from "lucide-react";

/**
 * BottomNav — Mobile-only bottom navigation bar.
 * Visible only on SM (<768px) via CSS media query.
 *
 * Shows all 7 sections in a horizontally scrollable strip.
 * This is the ONLY navigation on mobile — no hamburger menu.
 */
const BOTTOM_NAV_ITEMS = [
  { label: "Dashboard", route: "/dashboard", icon: BarChart3 },
  { label: "Dataset", route: "/dataset", icon: Database },
  { label: "Train", route: "/train", icon: BrainCircuit },
  { label: "Predict", route: "/predict", icon: Zap },
  { label: "Opt. Lab", route: "/optimization-lab", icon: FlaskConical },
  { label: "Interpret", route: "/interpret", icon: Eye },
  { label: "Settings", route: "/settings", icon: Settings },
] as const;

export default function BottomNav() {
  const pathname = usePathname();

  const isActive = (route: string) => {
    if (route === "/dashboard") return pathname === "/" || pathname === "/dashboard";
    return pathname.startsWith(route);
  };

  return (
    <nav className="bottom-nav" aria-label="Mobile navigation">
      {BOTTOM_NAV_ITEMS.map((item) => {
        const Icon = item.icon;
        const active = isActive(item.route);

        return (
          <Link
            key={item.route}
            href={item.route}
            className={`bottom-nav-item ${active ? "active" : ""}`}
            aria-current={active ? "page" : undefined}
          >
            <Icon size={20} strokeWidth={active ? 2.2 : 1.6} />
            <span className="bottom-nav-label">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
