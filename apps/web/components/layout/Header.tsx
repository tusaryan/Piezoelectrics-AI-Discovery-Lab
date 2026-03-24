"use client";

import * as React from "react";
import { useTheme } from "next-themes";
import { motion } from "framer-motion";
import { Moon, Sun, Menu } from "lucide-react";

interface HeaderProps {
  sidebarOpen: boolean;
  setSidebarOpen: (isOpen: boolean) => void;
  isMobile: boolean;
}

export function Header({ sidebarOpen, setSidebarOpen, isMobile }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => setMounted(true), []);

  return (
    <header className="sticky top-0 z-10 w-full h-16 border-b border-border bg-background/80 backdrop-blur-md flex items-center justify-between px-4 sm:px-6">
      <div className="flex items-center gap-4">
        {isMobile && (
          <button
            className="p-2 -ml-2 text-muted-foreground hover:text-foreground"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu size={20} />
          </button>
        )}
        
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
          <span className="text-sm font-medium text-muted-foreground hidden sm:inline-block">
            System Online
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {mounted && (
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="relative p-2 rounded-full overflow-hidden hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          >
            <span className="sr-only">Toggle theme</span>
            <div className="relative w-5 h-5">
              <motion.div
                initial={false}
                animate={{
                  scale: theme === 'dark' ? 1 : 0,
                  opacity: theme === 'dark' ? 1 : 0,
                  rotate: theme === 'dark' ? 0 : 90,
                }}
                transition={{ duration: 0.2, ease: "easeInOut" }}
                className="absolute inset-0"
              >
                <Moon size={20} />
              </motion.div>
              <motion.div
                initial={false}
                animate={{
                  scale: theme === 'light' ? 1 : 0,
                  opacity: theme === 'light' ? 1 : 0,
                  rotate: theme === 'light' ? 0 : -90,
                }}
                transition={{ duration: 0.2, ease: "easeInOut" }}
                className="absolute inset-0"
              >
                <Sun size={20} />
              </motion.div>
            </div>
          </button>
        )}
        
        <div className="h-8 w-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
          <span className="text-primary font-medium text-sm">US</span>
        </div>
      </div>
    </header>
  );
}
