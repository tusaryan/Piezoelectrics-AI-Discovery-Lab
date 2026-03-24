"use client";

import { motion } from "framer-motion";
import { Zap, Vibrate, Settings2 } from "lucide-react";

export type TrainingMode = "auto" | "compare" | "expert";

interface ModeSelectorProps {
  selected: TrainingMode;
  onSelect: (mode: TrainingMode) => void;
}

export function ModeSelector({ selected, onSelect }: ModeSelectorProps) {
  const modes = [
    {
      id: "auto",
      icon: Zap,
      title: "Auto-AI",
      desc: "Fully automated search, stacking ensembles.",
      color: "text-amber-500",
      bgClass: "from-amber-500/10 to-transparent",
      activeBorder: "border-amber-500",
      activeBg: "bg-amber-500/5",
    },
    {
      id: "compare",
      icon: Vibrate,
      title: "Compare",
      desc: "Run multiple models side-by-side easily.",
      color: "text-blue-500",
      bgClass: "from-blue-500/10 to-transparent",
      activeBorder: "border-blue-500",
      activeBg: "bg-blue-500/5",
    },
    {
      id: "expert",
      icon: Settings2,
      title: "Expert",
      desc: "Full manual control over hyperparameters.",
      color: "text-primary",
      bgClass: "from-primary/10 to-transparent",
      activeBorder: "border-primary",
      activeBg: "bg-primary/5",
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {modes.map((mode) => {
        const isActive = selected === mode.id;
        const Icon = mode.icon;
        
        return (
          <motion.div
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.98 }}
            key={mode.id}
            onClick={() => onSelect(mode.id as TrainingMode)}
            className={`relative overflow-hidden cursor-pointer rounded-xl border-2 p-6 transition-all duration-300 ${
              isActive ? `${mode.activeBorder} ${mode.activeBg}` : "border-muted-foreground/20 hover:border-muted-foreground/40 bg-card"
            }`}
          >
            {isActive && (
              <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${mode.bgClass} opacity-50`} />
            )}
            <div className="flex flex-col items-center text-center space-y-3">
              <div className={`p-3 rounded-full bg-background border ${isActive ? mode.color + " border-current/20 shadow-sm" : "text-muted-foreground border-transparent"}`}>
                <Icon className="w-6 h-6" />
              </div>
              <div>
                <h3 className={`font-semibold text-lg ${isActive ? "text-foreground" : "text-muted-foreground"}`}>
                  {mode.title}
                </h3>
                <p className="text-sm text-muted-foreground/80 mt-1 leading-tight">
                  {mode.desc}
                </p>
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
