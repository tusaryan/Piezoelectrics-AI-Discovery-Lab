"use client";

/**
 * InfoTooltip — Universal, viewport-aware info tooltip with drag handle.
 * 
 * Usage:
 *   <InfoTooltip text="Some help text" />
 * 
 * Features:
 * - Auto-positions to stay within viewport (flips top/bottom/left/right)
 * - Draggable via 6-dot grip handle for edge cases  
 * - Text selectable inside tooltip body
 * - Portal-rendered to escape overflow:hidden containers
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { createPortal } from "react-dom";
import { Info, GripVertical } from "lucide-react";

interface InfoTooltipProps {
  text: string;
  size?: number;
}

export default function InfoTooltip({ text, size = 11 }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const dragStart = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  const calcPosition = useCallback(() => {
    if (!btnRef.current) return;
    const rect = btnRef.current.getBoundingClientRect();
    const tooltipW = 280;
    const tooltipH = 120; // estimated
    const gap = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let top: number;
    let left: number;

    // Prefer above
    if (rect.top > tooltipH + gap) {
      top = rect.top - tooltipH - gap;
    } else {
      // Below
      top = rect.bottom + gap;
    }

    // Horizontal centering with clamp
    left = rect.left + rect.width / 2 - tooltipW / 2;
    if (left < 8) left = 8;
    if (left + tooltipW > vw - 8) left = vw - tooltipW - 8;

    // Clamp vertical
    if (top < 8) top = 8;
    if (top + tooltipH > vh - 8) top = vh - tooltipH - 8;

    setPos({ top, left });
    setDragOffset({ x: 0, y: 0 });
  }, []);

  const toggle = useCallback(() => {
    if (open) {
      setOpen(false);
      setPos(null);
    } else {
      calcPosition();
      setOpen(true);
    }
  }, [open, calcPosition]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        tooltipRef.current && !tooltipRef.current.contains(e.target as Node) &&
        btnRef.current && !btnRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
        setPos(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Drag handlers
  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY, ox: dragOffset.x, oy: dragOffset.y };
  }, [dragOffset]);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      if (!dragStart.current) return;
      setDragOffset({
        x: dragStart.current.ox + (e.clientX - dragStart.current.x),
        y: dragStart.current.oy + (e.clientY - dragStart.current.y),
      });
    };
    const onUp = () => {
      setDragging(false);
      dragStart.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [dragging]);

  // Recalculate position on scroll/resize
  useEffect(() => {
    if (!open) return;
    const recalc = () => calcPosition();
    window.addEventListener("resize", recalc);
    window.addEventListener("scroll", recalc, true);
    return () => {
      window.removeEventListener("resize", recalc);
      window.removeEventListener("scroll", recalc, true);
    };
  }, [open, calcPosition]);

  return (
    <>
      <button
        ref={btnRef}
        className="info-tooltip-trigger"
        onClick={toggle}
        title="More info"
        type="button"
      >
        <Info size={size} />
      </button>
      {open && pos && typeof document !== "undefined" && createPortal(
        <div
          ref={tooltipRef}
          className="info-tooltip-portal"
          style={{
            position: "fixed",
            top: pos.top + dragOffset.y,
            left: pos.left + dragOffset.x,
            zIndex: 9999,
          }}
        >
          <div
            className="info-tooltip-drag-handle"
            onMouseDown={onDragStart}
            title="Drag to reposition"
          >
            <GripVertical size={10} />
          </div>
          <div className="info-tooltip-body">
            {text}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
