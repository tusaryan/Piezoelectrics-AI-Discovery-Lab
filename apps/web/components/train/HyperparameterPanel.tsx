"use client";

import { useState } from "react";
import { Info } from "lucide-react";

interface ParamSchema {
  name: string;
  type: "int" | "float" | "float_log" | "bool" | "select" | "architecture_builder";
  default: any;
  min?: number;
  max?: number;
  label: string;
  help: string;
  beginner_tip: string;
  impact: "low" | "medium" | "high";
}

interface HyperparameterPanelProps {
  modelName: string;
  schema: ParamSchema[];
  values: Record<string, any>;
  onChange: (name: string, value: any) => void;
  disabled?: boolean;
}

export function HyperparameterPanel({ modelName, schema, values, onChange, disabled }: HyperparameterPanelProps) {
  if (!schema || schema.length === 0) {
    return (
      <div className="p-8 text-center border rounded-xl bg-card">
        <p className="text-muted-foreground mb-1">No hyperparameters available for {modelName}</p>
        <p className="text-sm text-muted-foreground/60">This model trains optimally with standard internal defaults.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border bg-card p-6 shadow-sm">
      <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
        {modelName} Configuration
      </h3>

      <div className="space-y-6">
        {schema.map((param) => {
          const val = values[param.name] ?? param.default;
          
          return (
            <div key={param.name} className="p-4 rounded-lg bg-background/50 border hover:border-primary/20 transition-colors">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full mt-1 shrink-0 ${
                    param.impact === 'high' ? 'bg-destructive' : 
                    param.impact === 'medium' ? 'bg-amber-500' : 'bg-muted-foreground/50'
                  }`} />
                  <div>
                    <div className="font-medium text-sm flex items-center gap-1.5">
                      {param.label}
                      <span className="text-xs bg-muted px-1.5 py-0.5 rounded text-muted-foreground font-mono">
                        {param.name}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">{param.help}</p>
                  </div>
                </div>
              </div>

              <div className="pl-4">
                {/* Dynamically render inputs based on schema type */}
                {(param.type === "int" || param.type === "float_log" || param.type === "float") && (
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={param.min}
                      max={param.max}
                      step={param.type === "int" ? 1 : (param.max! - param.min!) / 100}
                      value={val}
                      disabled={disabled}
                      onChange={(e) => onChange(param.name, param.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                    />
                    <input 
                      type="number" 
                      value={val}
                      disabled={disabled}
                      onChange={(e) => onChange(param.name, param.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                      className="w-20 rounded-md border bg-transparent px-2 py-1 text-sm text-right"
                    />
                  </div>
                )}
                
                {param.type === "bool" && (
                  <label className="flex items-center cursor-pointer">
                    <div className="relative">
                      <input 
                        type="checkbox" 
                        className="sr-only" 
                        checked={val as boolean}
                        onChange={(e) => onChange(param.name, e.target.checked)}
                        disabled={disabled}
                      />
                      <div className={`block w-10 h-6 rounded-full transition-colors ${val ? 'bg-primary' : 'bg-muted'}`}></div>
                      <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${val ? 'transform translate-x-4' : ''}`}></div>
                    </div>
                    <span className="ml-3 text-sm font-medium">{val ? "Enabled" : "Disabled"}</span>
                  </label>
                )}
              </div>
              
              <div className="mt-3 pl-4 text-xs flex items-start gap-1.5 text-indigo-500/80 dark:text-indigo-400/80">
                 <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                 <span>{param.beginner_tip}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
