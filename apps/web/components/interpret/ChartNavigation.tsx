"use client";

/**
 * ChartNavigation — SVG-level zoom/pan/reset/copy controls for chart containers.
 *
 * Wraps any chart content. On show: renders a floating toolbar at the bottom-left.
 * The toolbar adapts to the current theme (light/dark/night).
 *
 * Zoom/Pan work via CSS transform on the content wrapper.
 * Copy downloads the chart area as PNG via canvas.
 */

import { useEffect, useRef, useState, useCallback, type ReactNode } from "react";
import {
  Eye, EyeOff, ZoomIn, ZoomOut, Move, RotateCcw, Copy, Download,
} from "lucide-react";

interface ChartNavigationProps {
  children: ReactNode;
  containerRef: React.RefObject<HTMLDivElement | null>;
  id?: string;
}

type Theme = "light" | "dark" | "night";

interface ThemeColors {
  bg: string;
  border: string;
  text: string;
  textMuted: string;
  success: string;
  error: string;
}

const THEME_COLORS: Record<Theme, ThemeColors> = {
  light: {
    bg: "#ffffff",
    border: "#E4E4E7",
    text: "#1E1B4B",
    textMuted: "#6B6D8A",
    success: "#10B981",
    error: "#EF4444",
  },
  dark: {
    bg: "#1A1A2E",
    border: "#2D2D4A",
    text: "#E8E9FF",
    textMuted: "#6B6D8A",
    success: "#10B981",
    error: "#EF4444",
  },
  night: {
    bg: "#1C1917",
    border: "#292524",
    text: "#F0E6D3",
    textMuted: "#8A7F6E",
    success: "#10B981",
    error: "#EF4444",
  },
};

function getTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  const t = document.documentElement.getAttribute("data-theme");
  if (t === "light" || t === "dark" || t === "night") return t;
  return "dark";
}

export function ChartNavigation({ children, containerRef, id }: ChartNavigationProps) {
  const [visible, setVisible] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [theme, setTheme] = useState<Theme>("dark");
  const [copied, setCopied] = useState(false);

  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = getTheme();
    setTheme(t);
    const observer = new MutationObserver(() => {
      setTheme(getTheme());
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  // Compute chart colors from current theme
  const colors = THEME_COLORS[theme];

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

  const handleCopy = useCallback(async () => {
    const el = contentRef.current;
    if (!el) return;
    try {
      const svg = el.querySelector("svg");
      if (svg) {
        // Clone SVG and inline its styles for export
        const clone = svg.cloneNode(true) as SVGSVGElement;
        clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

        // Apply current colors as inline styles on the clone
        const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
        style.textContent = `
          * { color: ${colors.text}; fill: ${colors.text}; }
          text { fill: ${colors.textMuted}; font-family: monospace; }
          line, path { stroke: ${colors.border}; }
        `;
        clone.insertBefore(style, clone.firstChild);

        const serializer = new XMLSerializer();
        const svgStr = serializer.serializeToString(clone);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const img = new Image();
        const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(blob);

        img.onload = () => {
          canvas.width = img.width * 2;
          canvas.height = img.height * 2;
          ctx?.scale(2, 2);
          ctx?.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
          try {
            canvas.toBlob((blob) => {
              if (blob) {
                navigator.clipboard.write([
                  new ClipboardItem({ "image/png": blob }),
                ]);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }
            });
          } catch {
            // Clipboard API not available — try download fallback
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
    } catch {
      // Silently fail
    }
  }, [colors, id]);

  const handleDownload = useCallback(() => {
    const el = contentRef.current;
    if (!el) return;
    const svg = el.querySelector("svg");
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(clone);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
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

  return (
    <div className="chart-nav-root">
      {/* Content area */}
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

      {/* Toolbar — shown when visible, floating bottom-left */}
      {visible && (
        <div
          className="chart-nav-toolbar"
          style={{
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            color: colors.text,
          }}
        >
          {/* Hide */}
          <button
            className="chart-nav-btn"
            onClick={() => setHidden(true)}
            title="Hide chart"
            style={{ color: colors.textMuted }}
          >
            <EyeOff size={14} />
          </button>

          <div className="chart-nav-divider" style={{ background: colors.border }} />

          {/* Zoom in */}
          <button
            className="chart-nav-btn"
            onClick={handleZoomIn}
            title="Zoom in"
            style={{ color: scale >= 4 ? colors.textMuted : colors.text }}
            disabled={scale >= 4}
          >
            <ZoomIn size={14} />
          </button>

          {/* Zoom out */}
          <button
            className="chart-nav-btn"
            onClick={handleZoomOut}
            title="Zoom out"
            style={{ color: scale <= 0.3 ? colors.textMuted : colors.text }}
            disabled={scale <= 0.3}
          >
            <ZoomOut size={14} />
          </button>

          {/* Pan (move) */}
          <button
            className="chart-nav-btn"
            title="Drag to pan"
            style={{ color: colors.text }}
          >
            <Move size={14} />
          </button>

          {/* Reset */}
          <button
            className="chart-nav-btn"
            onClick={handleReset}
            title="Reset view"
            style={{ color: (scale !== 1 || offset.x !== 0 || offset.y !== 0) ? colors.success : colors.textMuted }}
          >
            <RotateCcw size={14} />
          </button>

          <div className="chart-nav-divider" style={{ background: colors.border }} />

          {/* Copy as PNG */}
          <button
            className="chart-nav-btn"
            onClick={handleCopy}
            title={copied ? "Copied!" : "Copy as PNG"}
            style={{ color: copied ? colors.success : colors.textMuted }}
          >
            {copied ? <Download size={14} /> : <Copy size={14} />}
          </button>

          {/* Download */}
          <button
            className="chart-nav-btn"
            onClick={handleDownload}
            title="Download as PNG"
            style={{ color: colors.textMuted }}
          >
            <Download size={14} />
          </button>
        </div>
      )}

      {/* Toggle visibility button — always shown in card actions */}
      <button
        className="chart-nav-toggle"
        onClick={() => setVisible((v) => !v)}
        title={visible ? "Hide chart controls" : "Show chart controls"}
        style={{
          background: colors.bg,
          border: `1px solid ${colors.border}`,
          color: visible ? colors.text : colors.textMuted,
        }}
      >
        <Eye size={14} />
      </button>
    </div>
  );
}