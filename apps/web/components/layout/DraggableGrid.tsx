"use client";

import React, { useState, useEffect, useCallback, useMemo, ReactNode } from "react";
import GridLayout, { type Layout, type LayoutItem } from "react-grid-layout";
import { useContainerWidth } from "react-grid-layout";
import { RotateCcw } from "lucide-react";
import { DraggableCard } from "./DraggableCard";
import { CardDrawer, HiddenCard } from "./CardDrawer";

import "react-grid-layout/css/styles.css";

const rglOverrides = `
.react-grid-item.react-grid-placeholder {
  background: hsl(var(--primary) / 0.15) !important;
  border: 2px dashed hsl(var(--primary) / 0.4) !important;
  border-radius: 12px !important;
  opacity: 1 !important;
}
.react-grid-item > .react-resizable-handle::after {
  border-color: hsl(var(--muted-foreground) / 0.4) !important;
}
.react-grid-item.react-draggable-dragging {
  z-index: 100 !important;
  box-shadow: 0 20px 40px rgba(0,0,0,0.15) !important;
  opacity: 0.9;
}
`;

export interface CardDefinition {
  key: string;
  title: string;
  icon?: ReactNode;
  defaultLayout: { x: number; y: number; w: number; h: number; minW?: number; minH?: number };
  component: ReactNode;
}

interface DraggableGridProps {
  pageKey: string;
  cards: CardDefinition[];
  cols?: number;
  rowHeight?: number;
}

function getStorageKey(pageKey: string) { return `piezo_layout_${pageKey}`; }
function getHiddenKey(pageKey: string) { return `piezo_hidden_${pageKey}`; }
function getCollapsedKey(pageKey: string) { return `piezo_collapsed_${pageKey}`; }

export function DraggableGrid({
  pageKey,
  cards,
  cols = 12,
  rowHeight = 80,
}: DraggableGridProps) {
  const { width, containerRef, mounted: widthMounted } = useContainerWidth();
  const [isMounted, setIsMounted] = useState(false);
  
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const [hiddenKeys, setHiddenKeys] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(localStorage.getItem(getHiddenKey(pageKey)) || "[]"); } catch { return []; }
  });
  const [collapsedKeys, setCollapsedKeys] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(localStorage.getItem(getCollapsedKey(pageKey)) || "[]"); } catch { return []; }
  });
  const [savedLayout, setSavedLayout] = useState<LayoutItem[] | null>(() => {
    if (typeof window === "undefined") return null;
    try { const sl = localStorage.getItem(getStorageKey(pageKey)); return sl ? JSON.parse(sl) : null; }
    catch { return null; }
  });

  // Track if a card is being dragged (for showing the hide drop zone)
  const [isDragging, setIsDragging] = useState(false);
  const [showHideZone, setShowHideZone] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(getHiddenKey(pageKey), JSON.stringify(hiddenKeys));
      localStorage.setItem(getCollapsedKey(pageKey), JSON.stringify(collapsedKeys));
    } catch { /* ignore */ }
  }, [hiddenKeys, collapsedKeys, pageKey]);

  const handleLayoutChange = useCallback(
    (newLayout: Layout) => {
      const items = newLayout as LayoutItem[];
      setSavedLayout(items);
      try { localStorage.setItem(getStorageKey(pageKey), JSON.stringify(items)); }
      catch { /* quota exceeded */ }
    },
    [pageKey]
  );

  const handleDragStart = useCallback(() => {
    setIsDragging(true);
    setShowHideZone(true);
  }, []);

  const handleDragStop = useCallback(() => {
    setIsDragging(false);
    // Delay hiding the zone so it persists briefly after drop
    setTimeout(() => setShowHideZone(false), 300);
  }, []);

  const handleCollapse = useCallback((key: string) => {
    setCollapsedKeys((prev) => prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]);
  }, []);

  const handleHide = useCallback((key: string) => {
    setHiddenKeys((prev) => [...prev, key]);
  }, []);

  const handleRestore = useCallback((key: string) => {
    setHiddenKeys((prev) => prev.filter((k) => k !== key));
  }, []);

  const handleReset = useCallback(() => {
    setSavedLayout(null);
    setHiddenKeys([]);
    setCollapsedKeys([]);
    localStorage.removeItem(getStorageKey(pageKey));
    localStorage.removeItem(getHiddenKey(pageKey));
    localStorage.removeItem(getCollapsedKey(pageKey));
  }, [pageKey]);

  // Handle drop from CardDrawer (restore card by dropping onto grid)
  const handleGridDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const cardKey = e.dataTransfer.getData("cardKey");
    if (cardKey) {
      handleRestore(cardKey);
    }
  }, [handleRestore]);

  const handleGridDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const visibleCards = useMemo(() => cards.filter((c) => !hiddenKeys.includes(c.key)), [cards, hiddenKeys]);

  const layout: LayoutItem[] = useMemo(() => {
    const defaultItems: LayoutItem[] = visibleCards.map((card) => ({
      i: card.key,
      x: card.defaultLayout.x,
      y: card.defaultLayout.y,
      w: card.defaultLayout.w,
      h: collapsedKeys.includes(card.key) ? 1 : card.defaultLayout.h,
      minW: card.defaultLayout.minW ?? 2,
      minH: collapsedKeys.includes(card.key) ? 1 : (card.defaultLayout.minH ?? 2),
    }));
    if (savedLayout) {
      return defaultItems.map((def) => {
        const saved = savedLayout.find((s) => s.i === def.i);
        return saved ? { ...saved, minW: def.minW, minH: def.minH } : def;
      });
    }
    return defaultItems;
  }, [visibleCards, savedLayout, collapsedKeys]);

  const hiddenCardsList: HiddenCard[] = useMemo(
    () => cards.filter((c) => hiddenKeys.includes(c.key)).map((c) => ({ key: c.key, title: c.title, icon: c.icon })),
    [cards, hiddenKeys]
  );

  if (!isMounted || !widthMounted) return null;

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: rglOverrides }} />
      <CardDrawer hiddenCards={hiddenCardsList} onRestore={handleRestore} />

      {/* Right-edge drop zone — appears when dragging a card */}
      {showHideZone && (
        <div
          className="fixed right-0 top-0 bottom-0 w-16 z-30 flex items-center justify-center transition-all duration-200"
          style={{ pointerEvents: isDragging ? "auto" : "none" }}
        >
          <div className="absolute inset-0 bg-gradient-to-l from-destructive/20 to-transparent" />
          <div className="relative text-destructive/60 text-xs font-semibold writing-mode-vertical flex flex-col items-center gap-1">
            <span className="transform -rotate-90 whitespace-nowrap">← Drag to hide</span>
          </div>
        </div>
      )}

      <div className="flex justify-end mb-2 px-2">
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground px-2.5 py-1.5 rounded-lg hover:bg-muted/60 transition-colors"
          title="Reset layout to default"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset Layout
        </button>
      </div>

      <div ref={containerRef} onDrop={handleGridDrop} onDragOver={handleGridDragOver}>
        <GridLayout
          width={width}
          layout={layout}
          gridConfig={{
            cols,
            rowHeight,
            margin: [16, 16] as readonly [number, number],
            containerPadding: [0, 0] as readonly [number, number],
            maxRows: Infinity,
          }}
          dragConfig={{
            enabled: true,
            handle: ".drag-handle",
          }}
          resizeConfig={{
            enabled: true,
          }}
          onLayoutChange={handleLayoutChange}
          onDragStart={handleDragStart}
          onDragStop={handleDragStop}
        >
          {visibleCards.map((card) => (
            <div key={card.key}>
              <DraggableCard
                title={card.title}
                icon={card.icon}
                cardKey={card.key}
                isCollapsed={collapsedKeys.includes(card.key)}
                onCollapse={handleCollapse}
                onHide={handleHide}
              >
                {card.component}
              </DraggableCard>
            </div>
          ))}
        </GridLayout>
      </div>
    </>
  );
}
