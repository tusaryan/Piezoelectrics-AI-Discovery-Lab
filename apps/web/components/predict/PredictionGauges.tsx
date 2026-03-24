"use client";

import { motion } from "framer-motion";
import { Activity, Thermometer } from "lucide-react";

interface PredictionGaugesProps {
  d33: number;
  tc: number;
  d33Lower: number;
  d33Upper: number;
  tcLower: number;
  tcUpper: number;
}

export function PredictionGauges({ d33, tc, d33Lower, d33Upper, tcLower, tcUpper }: PredictionGaugesProps) {
  // Normalize values for visual bars
  const maxD33 = 1000;
  const maxTc = 800;
  
  const d33Percent = Math.min(Math.max((d33 / maxD33) * 100, 5), 100);
  const tcPercent = Math.min(Math.max((tc / maxTc) * 100, 5), 100);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
      
      {/* d33 Gauge */}
      <div className="relative overflow-hidden rounded-2xl border bg-card p-6 shadow-sm group">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
        
        <div className="flex justify-between items-start mb-6">
          <div className="flex items-center gap-2 text-indigo-500">
            <Activity className="w-5 h-5" />
            <h3 className="font-semibold text-foreground tracking-tight">Piezoelectric Coefficient</h3>
          </div>
          <div className="text-xs font-mono bg-muted text-muted-foreground px-2 py-1 rounded">
            d₃₃ (pC/N)
          </div>
        </div>

        <div className="flex flex-col items-center">
          <motion.div 
            initial={{ scale: 0.5, opacity: 0 }} 
            animate={{ scale: 1, opacity: 1 }} 
            transition={{ type: "spring", bounce: 0.5 }}
            className="text-5xl font-bold tracking-tighter text-indigo-500 mb-1"
          >
            {d33.toFixed(1)}
          </motion.div>
          
          <div className="text-xs text-muted-foreground flex items-center gap-2 mb-8 mt-2">
            <span className="bg-muted px-1.5 py-0.5 rounded">95% CI</span>
            <span>[{d33Lower.toFixed(1)} - {d33Upper.toFixed(1)}]</span>
          </div>

          <div className="w-full h-3 bg-muted rounded-full overflow-hidden relative">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${d33Percent}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
              className="absolute top-0 left-0 bottom-0 bg-gradient-to-r from-indigo-400 to-indigo-600 rounded-full"
            />
          </div>
          <div className="w-full flex justify-between text-[10px] text-muted-foreground/60 mt-2 font-mono">
            <span>0</span>
            <span>500</span>
            <span>1000+</span>
          </div>
        </div>
      </div>

      {/* Tc Thermometer */}
      <div className="relative overflow-hidden rounded-2xl border bg-card p-6 shadow-sm group">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
        
        <div className="flex justify-between items-start mb-6">
          <div className="flex items-center gap-2 text-orange-500">
            <Thermometer className="w-5 h-5" />
            <h3 className="font-semibold text-foreground tracking-tight">Curie Temperature</h3>
          </div>
          <div className="text-xs font-mono bg-muted text-muted-foreground px-2 py-1 rounded">
            T_c (°C)
          </div>
        </div>

        <div className="flex flex-col items-center">
          <motion.div 
            initial={{ scale: 0.5, opacity: 0 }} 
            animate={{ scale: 1, opacity: 1 }} 
            transition={{ type: "spring", bounce: 0.5, delay: 0.1 }}
            className="text-5xl font-bold tracking-tighter text-orange-500 mb-1"
          >
            {tc.toFixed(1)}
          </motion.div>
          
          <div className="text-xs text-muted-foreground flex items-center gap-2 mb-8 mt-2">
            <span className="bg-muted px-1.5 py-0.5 rounded">95% CI</span>
            <span>[{tcLower.toFixed(1)} - {tcUpper.toFixed(1)}]</span>
          </div>

          <div className="w-full h-3 bg-muted rounded-full overflow-hidden relative">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${tcPercent}%` }}
              transition={{ duration: 1, ease: "easeOut", delay: 0.1 }}
              className="absolute top-0 left-0 bottom-0 bg-gradient-to-r from-orange-400 to-red-500 rounded-full"
            />
          </div>
          <div className="w-full flex justify-between text-[10px] text-muted-foreground/60 mt-2 font-mono">
            <span>0°C</span>
            <span>400°C</span>
            <span>800°C+</span>
          </div>
        </div>
      </div>

    </div>
  );
}
