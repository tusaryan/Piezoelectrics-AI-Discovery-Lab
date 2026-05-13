"use client";

/**
 * ChartNavigation — GitHub mermaid-style zoom/pan/reset/expand controls.
 *
 * Layout:
 * - Plus-like orientation with arrows: ← ↑ ↓ →
 * - Center: Reset (home) button
 * - Right-top: Zoom in (+)
 * - Right-bottom: Zoom out (-)
 * - Right edge, vertically: Expand (fullscreen) button
 * - Copy button with image/mermaid choice
 * - Hide/Unhide toggle
 * - Opaque background canvas behind the chart
 */

import { useEffect, useRef, useState, useCallback, type ReactNode } from "react";
import {
  Eye, EyeOff, ZoomIn, ZoomOut, RotateCcw, Move, Expand,
  Copy, Download, ChevronUp, ChevronDown, ChevronLeft, ChevronRight,
  Image, Code, X
} from "lucide-react";

interface ChartNavigationProps {
  children: ReactNode;
  containerRef: React.RefObject<HTMLDivElement | null>;
  id?: string;
  mermaidCode?: string; // Optional mermaid code for copy option
}

type Theme = "light" | "dark" | "night";

interface ThemeColors {
  bg: string;
  border: string;
  text: string;
  textMuted: string;
  success: string;
  error: string;
  canvasBg: string;
}

const THEME_COLORS: Record<Theme, ThemeColors> = {
  light: {
    bg: "#ffffff",
    border: "#D4D4D8",
    text: "#18181B",
    textMuted: "#71717A",
    success: "#22C55E",
    error: "#EF4444",
    canvasBg: "#FAFAFA",
  },
  dark: {
    bg: "#18181B",
    border: "#3F3F46",
    text: "#FAFAFA",
    textMuted: "#A1A1AA",
    success: "#22C55E",
    error: "#EF4444",
    canvasBg: "#09090B",
  },
  night: {
    bg: "#1C1917",
    border: "#44403C",
    text: "#F5F5F4",
    textMuted: "#A8A29E",
    success: "#22C55E",
    error: "#EF4444",
    canvasBg: "#1C1917",
  },
};

function getTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  const t = document.documentElement.getAttribute("data-theme");
  if (t === "light" || t === "dark" || t === "night") return t;
  return "dark";
}

export function ChartNavigation({ children, containerRef, id, mermaidCode }: ChartNavigationProps) {
  const [visible, setVisible] = useState(true);
  const [hidden, setHidden] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [theme, setTheme] = useState<Theme>("dark");
  const [copied, setCopied] = useState(false);
  const [showCopyMenu, setShowCopyMenu] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const contentRef = useRef<HTMLDivElement>(null);
  const expandedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = getTheme();
    setTheme(t);
    const observer = new MutationObserver(() => {
      setTheme(getTheme());
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  const colors = THEME_COLORS[theme];

  // Zoom handlers
  const handleZoomIn = useCallback(() => {
    setScale((s) => Math.min(s + 0.2, 4));
  }, []);

  const handleZoomOut = useCallback(() => {
    setScale((s) => Math.max(s - 0.2, 0.3));
  }, []);

  const handleReset = useCallback(() => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  // Pan handlers
  const handleMove = useCallback((direction: "up" | "down" | "left" | "right") => {
    const step = 50;
    setOffset((prev) => ({
      x: prev.x + (direction === "left" ? step : direction === "right" ? -step : 0),
      y: prev.y + (direction === "up" ? step : direction === "down" ? -step : 0),
    }));
  }, []);

  // Drag handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  }, [offset]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Copy as PNG
  const handleCopyImage = useCallback(async () => {
    const el = contentRef.current;
    if (!el) return;
    try {
      const svg = el.querySelector("svg");
      if (svg) {
        const clone = svg.cloneNode(true) as SVGSVGElement;
        clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        const styleEl = document.createElementNS("http://www.w3.org/2000/svg", "style");
        styleEl.textContent = `
          * { color: ${colors.text}; fill: ${colors.text}; }
          text { fill: ${colors.textMuted}; font-family: monospace; }
          line, path { stroke: ${colors.border}; }
        `;
        clone.insertBefore(styleEl, clone.firstChild);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const serializer = new (window as any).XMLSerializer();
        const svgStr = serializer.serializeToString(clone);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const img = new (window as any).Image();
        const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        img.onload = () => {
          canvas.width = img.width * 2;
          canvas.height = img.height * 2;
          ctx?.scale(2, 2);
          ctx?.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
          try {
            canvas.toBlob((b) => {
              if (b) {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                navigator.clipboard.write([new (window as any).ClipboardItem({ "image/png": b })]);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }
            });
          } catch {
            const a = document.createElement("a");
            a.href = canvas.toDataURL("image/png");
            a.download = `${id || "chart"}-${Date.now()}.png`;
            a.click();
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          }
        };
        img.src = url;
      }
    } catch { /* silent */ }
    setShowCopyMenu(false);
  }, [colors, id]);

  // Copy as Mermaid
  const handleCopyMermaid = useCallback(async () => {
    if (mermaidCode) {
      await navigator.clipboard.writeText(mermaidCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
    setShowCopyMenu(false);
  }, [mermaidCode]);

  // Download
  const handleDownload = useCallback(() => {
    const el = contentRef.current;
    if (!el) return;
    const svg = el.querySelector("svg");
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const serializer = new (window as any).XMLSerializer();
    const svgStr = serializer.serializeToString(clone);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const img = new (window as any).Image();
    const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      canvas.width = img.width * 2;
      canvas.height = img.height * 2;
      ctx?.scale(2, 2);
      ctx?.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = `${id || "chart"}-${Date.now()}.png`;
      a.click();
    };
    img.src = url;
  }, [id]);

  // Handle escape key for expanded mode
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isExpanded) {
        setIsExpanded(false);
      }
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [isExpanded]);

  if (hidden) {
    return (
      <div className="chart-nav-hidden-wrapper">
        <div
          className="chart-nav-hidden-backdrop"
          onClick={() => setHidden(false)}
          title="Click to show chart"
        >
          <Eye size={20} />
          <span>Chart hidden — click to show</span>
        </div>
      </div>
    );
  }

  // Expanded fullscreen view
  if (isExpanded) {
    return (
      <div
        className="chart-nav-expanded-overlay"
        onClick={(e) => {
          if (e.target === e.currentTarget) setIsExpanded(false);
        }}
      >
        <div className="chart-nav-expanded-container" ref={expandedRef}>
          <div className="chart-nav-expanded-header">
            <span style={{ color: colors.text }}>Expanded View</span>
            <button
              className="chart-nav-expanded-close"
              onClick={() => setIsExpanded(false)}
              style={{ color: colors.textMuted }}
            >
              <X size={20} />
            </button>
          </div>
          <div
            className="chart-nav-expanded-content"
            style={{
              background: colors.canvasBg,
            }}
          >
            {children}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-nav-root">
      {/* Opaque background canvas */}
      <div
        className="chart-nav-canvas"
        style={{ background: colors.canvasBg }}
      />

      {/* Content area with transform */}
      <div
        ref={contentRef}
        className="chart-nav-content"
        style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          transformOrigin: "center center",
          cursor: isDragging ? "grabbing" : scale > 1 ? "grab" : "default",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {children}
      </div>

      {/* Navigation controls - only show when visible */}
      {visible && (
        <>
          {/* Left side: Directional arrows + center reset */}
          <div className="chart-nav-dpad" style={{ background: colors.bg, borderColor: colors.border }}>
            {/* Up */}
            <button
              className="chart-nav-dpad-btn"
              onClick={() => handleMove("up")}
              title="Move up"
              style={{ color: colors.text }}
            >
              <ChevronUp size={16} />
            </button>

            {/* Left-Down-Right row */}
            <div className="chart-nav-dpad-row">
              <button
                className="chart-nav-dpad-btn"
                onClick={() => handleMove("left")}
                title="Move left"
                style={{ color: colors.text }}
              >
                <ChevronLeft size={16} />
              </button>

              {/* Center: Reset */}
              <button
                className="chart-nav-dpad-center"
                onClick={handleReset}
                title="Reset view"
                style={{
                  color: (scale !== 1 || offset.x !== 0 || offset.y !== 0) ? colors.success : colors.textMuted,
                  background: colors.canvasBg,
                }}
              >
                <RotateCcw size={14} />
              </button>

              <button
                className="chart-nav-dpad-btn"
                onClick={() => handleMove("right")}
                title="Move right"
                style={{ color: colors.text }}
              >
                <ChevronRight size={16} />
              </button>
            </div>

            {/* Down */}
            <button
              className="chart-nav-dpad-btn"
              onClick={() => handleMove("down")}
              title="Move down"
              style={{ color: colors.text }}
            >
              <ChevronDown size={16} />
            </button>
          </div>

          {/* Right side: Zoom controls + expand + copy */}
          <div className="chart-nav-zoom-panel" style={{ background: colors.bg, borderColor: colors.border }}>
            {/* Zoom in (top) */}
            <button
              className="chart-nav-zoom-btn"
              onClick={handleZoomIn}
              title="Zoom in"
              style={{ color: scale >= 4 ? colors.textMuted : colors.text }}
              disabled={scale >= 4}
            >
              <ZoomIn size={16} />
            </button>

            {/* Zoom level indicator */}
            <div className="chart-nav-zoom-level" style={{ color: colors.textMuted }}>
              {Math.round(scale * 100)}%
            </div>

            {/* Zoom out (bottom) */}
            <button
              className="chart-nav-zoom-btn"
              onClick={handleZoomOut}
              title="Zoom out"
              style={{ color: scale <= 0.3 ? colors.textMuted : colors.text }}
              disabled={scale <= 0.3}
            >
              <ZoomOut size={16} />
            </button>

            <div className="chart-nav-zoom-divider" style={{ background: colors.border }} />

            {/* Expand (fullscreen) */}
            <button
              className="chart-nav-zoom-btn"
              onClick={() => setIsExpanded(true)}
              title="Expand to fullscreen"
              style={{ color: colors.text }}
            >
              <Expand size={16} />
            </button>

            <div className="chart-nav-zoom-divider" style={{ background: colors.border }} />

            {/* Copy menu */}
            <div className="chart-nav-copy-wrapper">
              <button
                className="chart-nav-zoom-btn"
                onClick={() => setShowCopyMenu(!showCopyMenu)}
                title={copied ? "Copied!" : "Copy"}
                style={{ color: copied ? colors.success : colors.textMuted }}
              >
                {copied ? <Download size={16} /> : <Copy size={16} />}
              </button>

              {showCopyMenu && (
                <div className="chart-nav-copy-menu" style={{ background: colors.bg, borderColor: colors.border }}>
                  <button
                    className="chart-nav-copy-option"
                    onClick={handleCopyImage}
                    style={{ color: colors.text }}
                  >
                    <Image size={14} />
                    <span>Copy as Image</span>
                  </button>
                  {mermaidCode && (
                    <button
                      className="chart-nav-copy-option"
                      onClick={handleCopyMermaid}
                      style={{ color: colors.text }}
                    >
                      <Code size={14} />
                      <span>Copy as Mermaid</span>
                    </button>
                  )}
                  <button
                    className="chart-nav-copy-option"
                    onClick={handleDownload}
                    style={{ color: colors.text }}
                  >
                    <Download size={14} />
                    <span>Download PNG</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Hide button */}
          <button
            className="chart-nav-hide-btn"
            onClick={() => setHidden(true)}
            title="Hide controls"
            style={{ background: colors.bg, borderColor: colors.border, color: colors.textMuted }}
          >
            <EyeOff size={14} />
          </button>
        </>
      )}

      {/* Show toggle button when hidden controls */}
      {!visible && (
        <button
          className="chart-nav-show-btn"
          onClick={() => setVisible(true)}
          title="Show controls"
          style={{ background: colors.bg, borderColor: colors.border, color: colors.textMuted }}
        >
          <Eye size={14} />
        </button>
      )}
    </div>
  );
}