"use client";

import { CheckCircle2, XCircle, ShieldAlert } from "lucide-react";
import { motion } from "framer-motion";

interface ValidationResult {
  rule: string;
  expectation: string;
  observation: string;
}

interface PhysicsValidationProps {
  score: number;
  confirmed: ValidationResult[];
  violated: ValidationResult[];
}

export function PhysicsValidationCard({ score, confirmed, violated }: PhysicsValidationProps) {
  return (
    <div className="rounded-xl border bg-card overflow-hidden w-full">
       <div className="bg-muted/50 p-6 border-b flex justify-between items-center">
          <div>
            <h3 className="font-semibold text-lg flex items-center gap-2">
               <ShieldAlert className="w-5 h-5 text-indigo-500" />
               Physics Logic Validator
            </h3>
            <p className="text-sm text-muted-foreground mt-1">Checking learned SHAP associations against expected solid-state physics.</p>
          </div>
          
          <div className="flex flex-col items-end">
             <span className="text-4xl font-bold tracking-tighter" style={{ color: score > 80 ? '#10B981' : score > 50 ? '#F59E0B' : '#EF4444'}}>
                {score}%
             </span>
             <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Alignment Score</span>
          </div>
       </div>
       
       <div className="p-6 space-y-6">
          {violated.length > 0 && (
             <div>
                <h4 className="text-sm font-semibold text-rose-500 mb-3 uppercase tracking-wider">Violations Detected</h4>
                <div className="space-y-3">
                   {violated.map((v, i) => (
                      <motion.div initial={{opacity:0, x:20}} animate={{opacity:1, x:0}} transition={{delay: i*0.1}} key={i} className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-4">
                         <div className="flex items-start gap-3">
                            <XCircle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
                            <div>
                               <p className="font-semibold text-sm text-foreground">{v.rule}</p>
                               <p className="text-xs text-muted-foreground mt-1 mb-2"><span className="font-medium text-rose-400">Expected:</span> {v.expectation}</p>
                               <p className="text-xs text-muted-foreground"><span className="font-medium text-foreground">Observed:</span> {v.observation}</p>
                            </div>
                         </div>
                      </motion.div>
                   ))}
                </div>
             </div>
          )}

          {confirmed.length > 0 && (
             <div>
                <h4 className="text-sm font-semibold text-emerald-500 mb-3 uppercase tracking-wider">Confirmed Logic</h4>
                <div className="space-y-3">
                   {confirmed.map((c, i) => (
                      <motion.div initial={{opacity:0, x:-20}} animate={{opacity:1, x:0}} transition={{delay: i*0.1}} key={i} className="bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-4">
                         <div className="flex items-start gap-3">
                            <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                            <div>
                               <p className="font-semibold text-sm text-foreground">{c.rule}</p>
                               <p className="text-xs text-muted-foreground mt-1 mb-2"><span className="font-medium">Expected:</span> {c.expectation}</p>
                               <p className="text-xs text-muted-foreground"><span className="font-medium text-emerald-600/70">Observed:</span> {c.observation}</p>
                            </div>
                         </div>
                      </motion.div>
                   ))}
                </div>
             </div>
          )}
       </div>
    </div>
  );
}
