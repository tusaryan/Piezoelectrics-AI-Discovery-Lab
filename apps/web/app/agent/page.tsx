/**
 * Agent page — redirects to home since the agent is now a floating overlay.
 * This page remains for backwards compatibility with any bookmarks/links.
 */
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function AgentPage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to home — the floating agent is available on all pages
    router.replace("/");
  }, [router]);

  return (
    <div className="flex items-center justify-center h-64">
      <p className="text-muted-foreground text-sm">
        Redirecting... The AI Assistant is now available as a floating button on every page.
      </p>
    </div>
  );
}
