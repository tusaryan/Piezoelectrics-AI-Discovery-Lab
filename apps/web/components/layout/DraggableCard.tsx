"use client";

import { useState, useCallback, ReactNode } from "react";
import { GripVertical, Minimize2, Maximize2, Eye, EyeOff } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface DraggableCardProps {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  cardKey: string;
  isCollapsed?: boolean;
  onCollapse?: (key: string) => void;
  onHide?: (key: string) => void;
  className?: string;
}

export function DraggableCard({
  title,
  icon,
  children,
  cardKey,
  isCollapsed = false,
  onCollapse,
  onHide,
  className = "",
}: DraggableCardProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      className={`h-full flex flex-col rounded-xl border bg-card shadow-sm overflow-hidden transition-shadow ${
        hovered ? "shadow-lg ring-1 ring-primary/20" : ""
      } ${className}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Title bar — always visible */}
      <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/30 select-none min-h-[40px]">
        {/* Drag handle */}
        <div className="drag-handle cursor-grab active:cursor-grabbing p-1 rounded hover:bg-muted/60 transition-colors">
          <GripVertical className="w-4 h-4 text-muted-foreground/60" />
        </div>

        {/* Icon + Title */}
        {icon && <span className="shrink-0">{icon}</span>}
        <span className="text-sm font-semibold truncate flex-1">{title}</span>

        {/* Action buttons — show on hover */}
        <div
          className={`flex items-center gap-0.5 transition-opacity ${
            hovered ? "opacity-100" : "opacity-0"
          }`}
        >
          {onCollapse && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onCollapse(cardKey);
              }}
              className="p-1 rounded hover:bg-muted/60 transition-colors"
              title={isCollapsed ? "Expand" : "Collapse"}
            >
              {isCollapsed ? (
                <Maximize2 className="w-3.5 h-3.5 text-muted-foreground" />
              ) : (
                <Minimize2 className="w-3.5 h-3.5 text-muted-foreground" />
              )}
            </button>
          )}
          {onHide && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onHide(cardKey);
              }}
              className="p-1 rounded hover:bg-muted/60 transition-colors"
              title="Hide card"
            >
              <EyeOff className="w-3.5 h-3.5 text-muted-foreground" />
            </button>
          )}
        </div>
      </div>

      {/* Content — hidden when collapsed */}
      <AnimatePresence mode="wait">
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="flex-1 overflow-auto"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
