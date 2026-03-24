"use client";

import { motion } from "framer-motion";
import { Shield } from "lucide-react";

interface VickersGaugeProps {
  vickersHv: number;
}

export function VickersGauge({ vickersHv }: VickersGaugeProps) {
  const maxHv = 2000;
  const hvPercent = Math.min(Math.max((vickersHv / maxHv) * 100, 2), 100);

  return (
    <div className="relative overflow-hidden rounded-xl border bg-card p-6 shadow-sm group w-full">
      <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      
      <div className="flex justify-between items-start mb-6">
        <div className="flex items-center gap-2 text-rose-500">
          <Shield className="w-5 h-5" />
          <h3 className="font-semibold text-foreground tracking-tight">Vickers Microhardness</h3>
        </div>
        <div className="text-xs font-mono bg-muted text-muted-foreground px-2 py-1 rounded">
          HV (kgf/mm²)
        </div>
      </div>

      <div className="flex flex-col items-center">
        <motion.div 
          initial={{ scale: 0.5, opacity: 0 }} 
          animate={{ scale: 1, opacity: 1 }} 
          transition={{ type: "spring", bounce: 0.5 }}
          className="text-5xl font-bold tracking-tighter text-rose-500 mb-8 mt-2"
        >
          {vickersHv.toFixed(0)}
        </motion.div>

        <div className="w-full h-3 bg-muted rounded-full overflow-hidden relative">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${hvPercent}%` }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            className="absolute top-0 left-0 bottom-0 bg-gradient-to-r from-rose-400 to-rose-600 rounded-full"
          />
        </div>
        <div className="w-full flex justify-between text-[10px] text-muted-foreground/60 mt-2 font-mono">
          <span>0</span>
          <span>Polymer Matrix</span>
          <span>Hard PZT (~1000)</span>
          <span>Lead-Free High (2000+)</span>
        </div>
      </div>
    </div>
  );
}
