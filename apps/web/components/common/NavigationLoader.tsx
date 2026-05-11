"use client";

/**
 * NavigationLoader — Premium animated top-bar progress indicator.
 *
 * Listens to Next.js pathname changes and shows a thin, animated
 * gradient bar at the very top of the viewport during route transitions.
 * Theme-aware: uses CSS custom properties for colour.
 */

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { useUIStore } from "@/lib/store/uiStore";

export default function NavigationLoader() {
  const pathname = usePathname();
  const { isNavigating, setIsNavigating } = useUIStore();
  const prevPathRef = useRef(pathname);

  useEffect(() => {
    if (prevPathRef.current !== pathname) {
      // Route changed — show loader briefly
      setIsNavigating(true);
      prevPathRef.current = pathname;

      const timer = setTimeout(() => {
        setIsNavigating(false);
      }, 400);

      return () => clearTimeout(timer);
    }
  }, [pathname, setIsNavigating]);

  if (!isNavigating) return null;

  return (
    <div className="nav-loader" aria-hidden="true">
      <div className="nav-loader-bar" />
    </div>
  );
}
