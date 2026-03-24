"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BarChart3, Database, BrainCircuit, LineChart, 
  Layers, Diamond, Scale, Sparkles, Activity,
  Settings, ChevronLeft, ChevronRight, X 
} from "lucide-react";
import { features } from "@/lib/features";
import { cn } from "@/lib/utils";

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  isMobile: boolean;
}

export function Sidebar({ isOpen, setIsOpen, isMobile }: SidebarProps) {
  const pathname = usePathname();

  const navItems = [
    { name: "Dashboard", href: "/dashboard", icon: BarChart3, show: true },
    { name: "Dataset", href: "/dataset", icon: Database, show: true },
    { name: "Train", href: "/train", icon: BrainCircuit, show: true },
    { name: "Predict", href: "/predict", icon: LineChart, show: true },
    { name: "Composite", href: "/composite", icon: Layers, show: features.composite },
    { name: "Inverse Design", href: "/inverse", icon: Scale, show: true },
    { name: "Interpretability", href: "/interpret", icon: Sparkles, show: true },
    { name: "Active Learning", href: "/active-learning", icon: Activity, show: true },
    { name: "Hardness", href: "/hardness", icon: Diamond, show: true },
    { name: "Settings", href: "/settings", icon: Settings, show: true },
  ];

  const handleNavClick = () => {
    if (isMobile) {
      setIsOpen(false);
    }
  };

  // Mobile: overlay drawer
  if (isMobile) {
    return (
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
              onClick={() => setIsOpen(false)}
            />
            {/* Drawer */}
            <motion.div
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="fixed top-0 left-0 bottom-0 w-[280px] flex flex-col border-r border-border bg-card z-50 shadow-2xl"
            >
              <div className="flex items-center justify-between p-4 h-16 border-b border-border">
                <div className="font-bold text-xl text-primary flex items-center gap-2">
                  <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
                    <span className="text-primary-foreground text-xs leading-none">P</span>
                  </div>
                  Piezo.AI
                </div>
                <button 
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-md hover:bg-secondary text-muted-foreground transition-colors"
                >
                  <X size={18} />
                </button>
              </div>

              <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
                {navItems.filter(item => item.show).map((item) => {
                  const isActive = pathname.startsWith(item.href);
                  return (
                    <Link key={item.name} href={item.href} onClick={handleNavClick}>
                      <div
                        className={cn(
                          "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group relative",
                          isActive 
                            ? "bg-primary/10 text-primary font-medium" 
                            : "text-secondary-foreground hover:bg-secondary hover:text-foreground"
                        )}
                      >
                        {isActive && (
                          <motion.div 
                            layoutId="active-nav-indicator-mobile"
                            className="absolute left-0 w-1 h-full bg-primary rounded-r-full"
                            initial={false}
                            transition={{ type: "spring", stiffness: 300, damping: 30 }}
                          />
                        )}
                        <item.icon size={20} className={cn(
                          "flex-shrink-0 transition-colors",
                          isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                        )} />
                        <span className="truncate text-sm">{item.name}</span>
                      </div>
                    </Link>
                  );
                })}
              </nav>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    );
  }

  // Desktop: collapsible sidebar
  return (
    <motion.div
      initial={false}
      animate={{ width: isOpen ? 256 : 72 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="relative flex flex-col h-full border-r border-border bg-card z-20 flex-shrink-0 hide-scrollbar"
    >
      <div className="flex items-center justify-between p-4 h-16 border-b border-border">
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            transition={{ delay: 0.1 }}
            className="font-bold text-xl text-primary flex items-center gap-2"
          >
            <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
              <span className="text-primary-foreground text-xs leading-none">P</span>
            </div>
            Piezo.AI
          </motion.div>
        )}
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className={cn(
            "p-1.5 rounded-md hover:bg-secondary text-muted-foreground transition-colors",
            !isOpen && "mx-auto"
          )}
        >
          {isOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
        {navItems.filter(item => item.show).map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Link key={item.name} href={item.href} title={!isOpen ? item.name : undefined}>
              <div
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group relative",
                  isActive 
                    ? "bg-primary/10 text-primary font-medium" 
                    : "text-secondary-foreground hover:bg-secondary hover:text-foreground"
                )}
              >
                {isActive && (
                  <motion.div 
                    layoutId="active-nav-indicator"
                    className="absolute left-0 w-1 h-full bg-primary rounded-r-full"
                    initial={false}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                
                <item.icon 
                  size={20} 
                  className={cn(
                    "flex-shrink-0 transition-colors",
                    isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground",
                    !isOpen && "mx-auto"
                  )} 
                />
                
                {isOpen && (
                  <motion.span 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="truncate text-sm"
                  >
                    {item.name}
                  </motion.span>
                )}
              </div>
            </Link>
          );
        })}
      </nav>
    </motion.div>
  );
}
