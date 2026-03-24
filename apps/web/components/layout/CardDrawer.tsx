"use client";

import { useState, ReactNode } from "react";
import { ChevronLeft, ChevronRight, Eye, GripVertical } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export interface HiddenCard {
  key: string;
  title: string;
  icon?: ReactNode;
}

interface CardDrawerProps {
  hiddenCards: HiddenCard[];
  onRestore: (key: string) => void;
}

export function CardDrawer({ hiddenCards, onRestore }: CardDrawerProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (hiddenCards.length === 0 && !isOpen) return null;

  return (
    <>
      {/* Hover trigger strip on RIGHT edge */}
      {!isOpen && hiddenCards.length > 0 && (
        <div
          className="fixed right-0 top-1/2 -translate-y-1/2 z-50 group"
          onClick={() => setIsOpen(true)}
        >
          <div className="flex items-center gap-0 cursor-pointer flex-row-reverse">
            <div className="w-1.5 h-20 bg-primary/30 rounded-l-full group-hover:bg-primary/60 group-hover:w-2 transition-all" />
            <motion.div
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-card border border-r-0 rounded-l-lg shadow-lg p-1.5 -mr-px group-hover:translate-x-0 translate-x-1 transition-transform relative"
            >
              <ChevronLeft className="w-4 h-4 text-primary" />
              {hiddenCards.length > 0 && (
                <span className="absolute -top-1.5 -left-1.5 w-4 h-4 rounded-full bg-primary text-primary-foreground text-[10px] font-bold flex items-center justify-center">
                  {hiddenCards.length}
                </span>
              )}
            </motion.div>
          </div>
        </div>
      )}

      {/* Drawer panel — slides from RIGHT */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Drawer */}
            <motion.div
              initial={{ x: 280 }}
              animate={{ x: 0 }}
              exit={{ x: 280 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="fixed right-0 top-0 bottom-0 w-[260px] bg-card border-l shadow-2xl z-50 flex flex-col"
            >
              <div className="flex items-center justify-between px-4 py-3 border-b">
                <h3 className="text-sm font-semibold">Hidden Cards</h3>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1 rounded hover:bg-muted transition-colors"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              <div className="flex-1 overflow-auto p-3 space-y-2">
                {hiddenCards.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-8">
                    No hidden cards. Use the eye icon on any card to hide it here.
                  </p>
                ) : (
                  hiddenCards.map((card) => (
                    <div
                      key={card.key}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData("cardKey", card.key);
                        e.dataTransfer.effectAllowed = "move";
                      }}
                      className="flex items-center gap-2 p-2.5 rounded-lg border bg-muted/20 hover:bg-muted/40 transition-colors group cursor-grab active:cursor-grabbing"
                      onClick={() => {
                        onRestore(card.key);
                        if (hiddenCards.length <= 1) setIsOpen(false);
                      }}
                    >
                      <GripVertical className="w-3.5 h-3.5 text-muted-foreground/50 group-hover:text-muted-foreground transition-colors" />
                      {card.icon && <span className="shrink-0">{card.icon}</span>}
                      <span className="text-sm truncate flex-1">{card.title}</span>
                      <Eye className="w-3.5 h-3.5 text-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  ))
                )}
              </div>

              <div className="px-4 py-2 border-t text-[10px] text-muted-foreground text-center">
                Click or drag cards to restore them
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
