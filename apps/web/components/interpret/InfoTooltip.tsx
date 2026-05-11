"use client";

/**
 * InfoTooltip — reusable info tooltip for chart explanations.
 */

import { useState, useRef, useEffect } from "react";
import { Info, X } from "lucide-react";

interface InfoTooltipProps {
  title: string;
  children: React.ReactNode;
}

export default function InfoTooltip({ title, children }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="info-tooltip-wrapper" ref={ref}>
      <button
        className="info-tooltip-trigger"
        onClick={() => setOpen(!open)}
        aria-label={`Info: ${title}`}
      >
        <Info size={14} />
      </button>
      {open && (
        <div className="info-tooltip-popup">
          <div className="info-tooltip-header">
            <span>{title}</span>
            <button onClick={() => setOpen(false)}>
              <X size={12} />
            </button>
          </div>
          <div className="info-tooltip-body">{children}</div>
        </div>
      )}
    </div>
  );
}
