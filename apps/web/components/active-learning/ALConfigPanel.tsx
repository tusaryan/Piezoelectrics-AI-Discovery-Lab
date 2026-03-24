"use client";

import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Play, Settings2 } from "lucide-react";

interface ALConfigPanelProps {
  onStart: (config: any) => void;
  isRunning: boolean;
}

export function ALConfigPanel({ onStart, isRunning }: ALConfigPanelProps) {
  const [budget, setBudget] = useState(50);
  const [initSize, setInitSize] = useState(20);
  const [strategy, setStrategy] = useState("UCB");

  return (
    <div className="bg-card border rounded-xl shadow-sm p-6 space-y-8">
       <div className="flex items-center gap-3 border-b pb-4">
          <div className="p-2 bg-indigo-500/10 text-indigo-500 rounded-lg">
             <Settings2 className="w-5 h-5" />
          </div>
          <h2 className="text-lg font-semibold">Simulation Config</h2>
       </div>

       <div className="space-y-6">
          {/* Budget */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-sm font-medium">Experimental Budget</Label>
                <span className="text-sm text-muted-foreground font-mono">{budget} iter</span>
             </div>
             <Slider 
                value={[budget]} 
                onValueChange={(v) => setBudget(v[0])} 
                min={20} max={200} step={10}
                disabled={isRunning}
             />
          </div>

          {/* Initial Dataset */}
          <div className="space-y-3">
             <div className="flex justify-between">
                <Label className="text-sm font-medium">Initial Knowledge Base</Label>
                <span className="text-sm text-muted-foreground font-mono">{initSize} pts</span>
             </div>
             <Slider 
                value={[initSize]} 
                onValueChange={(v) => setInitSize(v[0])} 
                min={5} max={50} step={5}
                disabled={isRunning}
             />
          </div>

          {/* Strategy */}
          <div className="space-y-3">
             <Label className="text-sm font-medium">Acquisition Strategy</Label>
             <RadioGroup value={strategy} onValueChange={setStrategy} disabled={isRunning} className="gap-3 mt-2">
                <Label className={`flex items-center justify-between border p-3 rounded-lg cursor-pointer hover:bg-muted/30 transition-colors ${strategy === 'UCB' ? 'border-indigo-500 bg-indigo-500/5' : ''}`}>
                   <div className="flex items-center gap-3">
                      <RadioGroupItem value="UCB" />
                      <span className="font-medium">UCB (Upper Confidence Bound)</span>
                   </div>
                   <span className="text-xs text-muted-foreground">Balances explore/exploit</span>
                </Label>
                <Label className={`flex items-center justify-between border p-3 rounded-lg cursor-pointer hover:bg-muted/30 transition-colors ${strategy === 'EI' ? 'border-indigo-500 bg-indigo-500/5' : ''}`}>
                   <div className="flex items-center gap-3">
                      <RadioGroupItem value="EI" />
                      <span className="font-medium">EI (Expected Improvement)</span>
                   </div>
                   <span className="text-xs text-muted-foreground">Greedy optimization</span>
                </Label>
             </RadioGroup>
          </div>

          <Button 
             onClick={() => onStart({ budget, initSize, strategy })}
             disabled={isRunning}
             className="w-full h-12 text-base shadow-sm group bg-indigo-600 hover:bg-indigo-700 text-white"
          >
             {isRunning ? (
                <>
                   <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
                   Simulating Lab...
                </>
             ) : (
                <>
                   <Play className="w-5 h-5 mr-2 fill-current opacity-80 group-hover:opacity-100 transition-opacity" />
                   Start Active Learning
                </>
             )}
          </Button>
       </div>
    </div>
  );
}
