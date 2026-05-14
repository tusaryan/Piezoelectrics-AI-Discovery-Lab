"use client";

import {
  useState,
  useRef,
  useCallback,
  type ReactNode,
  type CSSProperties,
} from "react";
import {
  ChevronUp,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  Copy,
  Download,
  Eye,
  EyeOff,
  X,
  Image as ImageIcon,
  Code,
} from "lucide-react";

interface ChartNavigatorProps {
  children: ReactNode;
  /** Unique id for the chart container */
  chartId?: string;
  /** Enable/disable the navigator controls */
  enabled?: boolean;
  /** Optional minimum height for the chart viewport */
  minHeight?: number;
  /** Called when copy as image is requested */
  onCopyImage?: () => void;
  /** Called when download is requested */
  onDownload?: () => void;
  /** Optional mermaid/source code for "copy code" action */
  sourceCode?: string;
}

const PAN_STEP = 60;
const ZOOM_STEP = 0.15;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 3.0;

export default function ChartNavigator({
  children,
  chartId,
  enabled = true,
  minHeight = 260,
  onCopyImage,
  onDownload,
  sourceCode,
}: ChartNavigatorProps) {
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [expanded, setExpanded] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [showCopyMenu, setShowCopyMenu] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Pan handlers
  const panUp = useCallback(() => setPan((p) => ({ ...p, y: p.y + PAN_STEP })), []);
  const panDown = useCallback(() => setPan((p) => ({ ...p, y: p.y - PAN_STEP })), []);
  const panLeft = useCallback(() => setPan((p) => ({ ...p, x: p.x + PAN_STEP })), []);
  const panRight = useCallback(() => setPan((p) => ({ ...p, x: p.x - PAN_STEP })), []);
  const resetView = useCallback(() => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  }, []);

  // Zoom handlers
  const zoomIn = useCallback(
    () => setZoom((z) => Math.min(z + ZOOM_STEP, ZOOM_MAX)),
    [],
  );
  const zoomOut = useCallback(
    () => setZoom((z) => Math.max(z - ZOOM_STEP, ZOOM_MIN)),
    [],
  );

  // Expand/collapse
  const toggleExpand = useCallback(() => {
    setExpanded((e) => !e);
    if (expanded) resetView();
  }, [expanded, resetView]);

  // Snapshot: find SVG inside content, render to canvas, then blob
  const captureSnapshot = useCallback(async (): Promise<Blob | null> => {
    const el = contentRef.current;
    if (!el) return null;
    const svg = el.querySelector("svg");
    if (svg) {
      // Clone SVG to isolate rendering
      const clone = svg.cloneNode(true) as SVGSVGElement;
      const { width, height } = svg.getBoundingClientRect();
      clone.setAttribute("width", String(width * 2));
      clone.setAttribute("height", String(height * 2));
      clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
      // Inline computed styles for cross-context rendering
      const data = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([data], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new window.Image();
      img.crossOrigin = "anonymous";
      return new Promise<Blob | null>((resolve) => {
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = width * 2;
          canvas.height = height * 2;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            // Fill with theme background
            const bg = getComputedStyle(document.documentElement)
              .getPropertyValue("--card")
              .trim();
            ctx.fillStyle = bg || "#15162A";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
          }
          canvas.toBlob((b) => resolve(b), "image/png");
          URL.revokeObjectURL(url);
        };
        img.onerror = () => {
          URL.revokeObjectURL(url);
          resolve(null);
        };
        img.src = url;
      });
    }
    return null;
  }, []);

  // Copy as image
  const handleCopyImage = useCallback(async () => {
    if (onCopyImage) {
      onCopyImage();
      setShowCopyMenu(false);
      return;
    }
    try {
      const blob = await captureSnapshot();
      if (blob) {
        try {
          await navigator.clipboard.write([
            new ClipboardItem({ "image/png": blob }),
          ]);
          setCopyFeedback("Copied!");
        } catch {
          // Fallback: download
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `${chartId || "chart"}.png`;
          a.click();
          URL.revokeObjectURL(url);
          setCopyFeedback("Downloaded!");
        }
      } else {
        setCopyFeedback("No chart found");
      }
    } catch {
      setCopyFeedback("Failed");
    }
    setTimeout(() => setCopyFeedback(""), 1500);
    setShowCopyMenu(false);
  }, [onCopyImage, chartId, captureSnapshot]);

  // Copy source code
  const handleCopyCode = useCallback(async () => {
    if (!sourceCode) return;
    try {
      await navigator.clipboard.writeText(sourceCode);
      setCopyFeedback("Code copied!");
    } catch {
      setCopyFeedback("Failed");
    }
    setTimeout(() => setCopyFeedback(""), 1500);
    setShowCopyMenu(false);
  }, [sourceCode]);

  // Download as image
  const handleDownload = useCallback(async () => {
    if (onDownload) {
      onDownload();
      setShowCopyMenu(false);
      return;
    }
    try {
      const blob = await captureSnapshot();
      if (blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${chartId || "chart"}.png`;
        a.click();
        URL.revokeObjectURL(url);
        setCopyFeedback("Downloaded!");
      } else {
        setCopyFeedback("No chart found");
      }
    } catch {
      setCopyFeedback("Failed");
    }
    setTimeout(() => setCopyFeedback(""), 1500);
    setShowCopyMenu(false);
  }, [onDownload, chartId, captureSnapshot]);

  if (!enabled) {
    return <>{children}</>;
  }

  const transformStyle: CSSProperties = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    transformOrigin: "center center",
    transition: "transform 0.2s ease",
  };

  return (
    <div
      className={`cn-wrapper ${expanded ? "cn-expanded" : ""}`}
      ref={containerRef}
      data-chart-id={chartId}
    >
      {/* Chart viewport */}
      <div
        className="cn-viewport"
        style={{ minHeight: expanded ? "80vh" : minHeight }}
      >
        <div className="cn-content" ref={contentRef} style={transformStyle}>
          {children}
        </div>
      </div>

      {/* Toolbar: top-right of chart */}
      <div className="cn-toolbar">
        <button
          className="cn-toolbar-btn"
          onClick={toggleExpand}
          title={expanded ? "Collapse" : "Expand"}
        >
          {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </button>

        {/* Copy/Download dropdown */}
        <div className="cn-copy-dropdown-wrapper">
          <button
            className="cn-toolbar-btn"
            onClick={() => setShowCopyMenu(!showCopyMenu)}
            title="Copy / Download"
          >
            <Copy size={14} />
          </button>
          {showCopyMenu && (
            <div className="cn-copy-menu">
              <button onClick={handleCopyImage}>
                <ImageIcon size={13} /> Copy as Image
              </button>
              <button onClick={handleDownload}>
                <Download size={13} /> Download PNG
              </button>
              {sourceCode && (
                <button onClick={handleCopyCode}>
                  <Code size={13} /> Copy Code
                </button>
              )}
            </div>
          )}
        </div>

        {/* Hide/Show controls */}
        <button
          className="cn-toolbar-btn"
          onClick={() => setControlsVisible(!controlsVisible)}
          title={controlsVisible ? "Hide controls" : "Show controls"}
        >
          {controlsVisible ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      </div>

      {/* Copy feedback toast */}
      {copyFeedback && <div className="cn-toast">{copyFeedback}</div>}

      {/* D-pad + Zoom controls */}
      {controlsVisible && (
        <div className="cn-controls">
          {/* Row 1: Up + ZoomIn */}
          <div className="cn-row">
            <div className="cn-spacer" />
            <button className="cn-btn" onClick={panUp} title="Pan up">
              <ChevronUp size={14} />
            </button>
            <button className="cn-btn" onClick={zoomIn} title="Zoom in">
              <ZoomIn size={14} />
            </button>
          </div>
          {/* Row 2: Left + Reset + Right */}
          <div className="cn-row">
            <button className="cn-btn" onClick={panLeft} title="Pan left">
              <ChevronLeft size={14} />
            </button>
            <button
              className="cn-btn cn-reset"
              onClick={resetView}
              title="Reset view"
            >
              <RotateCcw size={13} />
            </button>
            <button className="cn-btn" onClick={panRight} title="Pan right">
              <ChevronRight size={14} />
            </button>
          </div>
          {/* Row 3: Down + ZoomOut */}
          <div className="cn-row">
            <div className="cn-spacer" />
            <button className="cn-btn" onClick={panDown} title="Pan down">
              <ChevronDown size={14} />
            </button>
            <button className="cn-btn" onClick={zoomOut} title="Zoom out">
              <ZoomOut size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Close button when expanded */}
      {expanded && (
        <button className="cn-close" onClick={toggleExpand} title="Close">
          <X size={18} />
        </button>
      )}
    </div>
  );
}
