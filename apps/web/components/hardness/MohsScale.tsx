"use client";

import { motion } from "framer-motion";

interface MohsScaleProps {
  predictedMohs: number;
}

const MINERALS = [
  { val: 1, name: "Talc", color: "bg-emerald-500", labelText: "text-emerald-600" },
  { val: 2, name: "Gypsum", color: "bg-emerald-500", labelText: "text-emerald-700" },
  { val: 3, name: "Calcite", color: "bg-teal-500", labelText: "text-teal-700" },
  { val: 4, name: "Fluorite", color: "bg-cyan-500", labelText: "text-cyan-700" },
  { val: 5, name: "Apatite", color: "bg-blue-500", labelText: "text-blue-700" },
  { val: 6, name: "Orthoclase", color: "bg-indigo-500", labelText: "text-indigo-700" },
  { val: 7, name: "Quartz", color: "bg-violet-500", labelText: "text-violet-700" },
  { val: 8, name: "Topaz", color: "bg-purple-500", labelText: "text-purple-700" },
  { val: 9, name: "Corundum", color: "bg-fuchsia-500", labelText: "text-fuchsia-700" },
  { val: 10, name: "Diamond", color: "bg-rose-500", labelText: "text-rose-700" },
];

export function MohsScale({ predictedMohs }: MohsScaleProps) {
  return (
    <div className="rounded-xl border bg-card p-6 shadow-sm w-full relative">
      <h3 className="font-semibold text-lg mb-4">Mohs Hardness Scale</h3>
      
      <div className="w-full overflow-x-auto pb-10 scrollbar-hide -mx-2 px-2">
        <div className="relative pt-6 min-w-[450px]">
          {/* Track */}
          <div className="absolute top-8 left-4 right-4 h-2 bg-gradient-to-r from-emerald-400 via-blue-500 to-rose-500 rounded-full opacity-30" />
          
          {/* Indicator */}
          <motion.div 
            initial={{ left: "0%" }}
            animate={{ left: `${((Math.max(1, Math.min(10, predictedMohs)) - 1) / 9) * 100}%` }}
            transition={{ type: "spring", bounce: 0.4, duration: 2 }}
            className="absolute top-2 w-0 h-0 border-l-[8px] border-r-[8px] border-t-[12px] border-l-transparent border-r-transparent border-t-foreground -ml-2 z-10"
          >
            <div className="absolute -top-10 -left-6 bg-foreground text-background text-xs font-bold px-2 py-1 rounded shadow-lg whitespace-nowrap">
              {predictedMohs.toFixed(1)} Mohs
            </div>
          </motion.div>

          {/* Minerals List */}
          <div className="flex justify-between relative z-0 mt-4 px-2">
            {MINERALS.map((m) => (
              <div key={m.val} className="flex flex-col items-center">
                <div className={`w-3 h-3 rounded-full border-2 border-background shadow-sm ${m.color}`} />
                <span className="text-[10px] text-muted-foreground mt-3 font-medium -rotate-45 origin-top-left translate-y-1 whitespace-nowrap">
                  {m.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
