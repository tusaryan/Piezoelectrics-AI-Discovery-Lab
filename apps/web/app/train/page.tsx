"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Play, Square, Plus, Trash2, Database, BrainCircuit, Activity, Network,
  ChevronUp, ChevronDown, Pencil
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { LogTerminal } from "@/components/train/LogTerminal";
import { HyperparameterPanel } from "@/components/train/HyperparameterPanel";
import { TrainingLossChart } from "@/components/train/TrainingLossChart";
import { ModelComparisonDashboard } from "@/components/train/ModelComparisonDashboard";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

type JobConfig = {
  id: string;
  targets: string[];
  model: string;
  use_optuna: boolean;
  optuna_trials: number;
  params: Record<string, any>;
};

const AVAILABLE_TARGETS = [
  { id: "d33", label: "Piezoelectric Constant (d33)" },
  { id: "tc", label: "Curie Temperature (Tc)" },
  { id: "vickers_hardness", label: "Vickers Hardness (HV)" }
];
const AVAILABLE_MODELS = ["XGBoost", "RandomForest", "GradientBoosting", "LightGBM", "SVR", "DecisionTree", "GPR", "ANN", "Stacking"];

export default function TrainPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [schema, setSchema] = useState<any>(null);
  const [artifacts, setArtifacts] = useState<any[]>([]);

  // 1. Top Section State (Currently Building Pipeline)
  const [pendingConfig, setPendingConfig] = useState<Omit<JobConfig, "id">>({
    targets: ["d33"],
    model: "XGBoost",
    use_optuna: false,
    optuna_trials: 50,
    params: {}
  });

  // 2. Bottom Section State (Queued Pipelines executing map)
  const [jobsQueue, setJobsQueue] = useState<JobConfig[]>([]);

  // 3. Runtime State
  const [isTraining, setIsTraining] = useState(false);
  const [activeJobIds, setActiveJobIds] = useState<string[]>([]);
  const [activeJobLogTarget, setActiveJobLogTarget] = useState<string | null>(null);
  
  // Real extracted metrics for graph [ {epoch, loss} ]
  const [graphData, setGraphData] = useState<{ epoch: number, loss: number }[]>([]);

  const fetchArtifacts = async (dsId: string) => {
    if (!dsId) return;
    try {
      const res = await fetch(`/api/v1/training/artifacts/${dsId}`);
      if (res.ok) {
        const d = await res.json();
        setArtifacts(d.data || []);
      }
    } catch(e) { console.error(e); }
  };

  useEffect(() => {
    fetch("/api/v1/datasets")
      .then(r => r.json())
      .then(d => {
        if (d.data?.length > 0) {
          setDatasets(d.data);
          setSelectedDatasetId(d.data[0].id);
        }
      });
    fetch("/api/v1/training/model-schema")
      .then(r => r.json())
      .then(d => setSchema(d.data));
  }, []);

  useEffect(() => {
    fetchArtifacts(selectedDatasetId);
  }, [selectedDatasetId]);

  const handleAddPipeline = () => {
    if (pendingConfig.targets.length === 0) return;
    setJobsQueue(prev => [...prev, {
      id: Math.random().toString(36).substring(7),
      targets: [...pendingConfig.targets],
      model: pendingConfig.model,
      use_optuna: pendingConfig.use_optuna,
      optuna_trials: pendingConfig.optuna_trials,
      params: { ...pendingConfig.params }
    }]);
    setPendingConfig({
      targets: ["tc"], 
      model: "RandomForest",
      use_optuna: false,
      optuna_trials: 50,
      params: {}
    });
  };

  const handleEditPipeline = (conf: JobConfig) => {
    setPendingConfig({
      targets: [...conf.targets],
      model: conf.model,
      use_optuna: conf.use_optuna,
      optuna_trials: conf.optuna_trials,
      params: { ...conf.params }
    });
    setJobsQueue(prev => prev.filter(c => c.id !== conf.id));
  };

  const handleRemovePipeline = (id: string) => {
    setJobsQueue(prev => prev.filter(c => c.id !== id));
  };

  const handleMoveUp = (index: number) => {
    if (index === 0) return;
    setJobsQueue(prev => {
      const copy = [...prev];
      const temp = copy[index];
      copy[index] = copy[index - 1];
      copy[index - 1] = temp;
      return copy;
    });
  };

  const handleMoveDown = (index: number) => {
    if (index === jobsQueue.length - 1) return;
    setJobsQueue(prev => {
      const copy = [...prev];
      const temp = copy[index];
      copy[index] = copy[index + 1];
      copy[index + 1] = temp;
      return copy;
    });
  };

  const startTraining = async () => {
    if (!selectedDatasetId || jobsQueue.length === 0) return;
    
    setIsTraining(true);
    setGraphData([]); 
    
    const launchedIds: string[] = [];
    for (const conf of jobsQueue) {
      if (conf.targets.length === 0) continue;
      for (const tgt of conf.targets) {
        try {
          const res = await fetch("/api/v1/training/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              dataset_id: selectedDatasetId,
              target: tgt,
              model_name: conf.model,
              mode: "expert",
              params: conf.params,
              use_optuna: conf.use_optuna,
              optuna_trials: conf.optuna_trials
            })
          });
          const data = await res.json();
          if (data.job_id) {
            launchedIds.push(data.job_id);
          }
        } catch (err) {
          console.error("Failed to start job", err);
        }
      }
    }
    
    setActiveJobIds(launchedIds);
    if (launchedIds.length > 0) {
      setActiveJobLogTarget(launchedIds[0]); 
    }
  };

  const stopTraining = async () => {
    for (const jid of activeJobIds) {
      try {
        await fetch(`/api/v1/training/${jid}/cancel`, { method: "POST" });
      } catch(e) { console.error("Cancel failed", e); }
    }
    setIsTraining(false);
    setActiveJobLogTarget(null);
    fetchArtifacts(selectedDatasetId);
  };

  const handleEpochLoss = useCallback((epoch: number, loss: number) => {
     setGraphData(prev => {
        const newData = [...prev, { epoch, loss }];
        if (newData.length > 150) return newData.slice(newData.length - 150);
        return newData;
     });
  }, []);

  const onJobComplete = useCallback(() => {
    setIsTraining(false);
    fetchArtifacts(selectedDatasetId);
  }, [selectedDatasetId]);

  const cards: CardDefinition[] = [
    {
      key: "configurator",
      title: "Pipeline Configurator",
      icon: <Database className="w-4 h-4 text-primary" />,
      defaultLayout: { x: 0, y: 0, w: 5, h: 7, minW: 4, minH: 4 },
      component: (
        <div className="p-4 space-y-4 overflow-auto">

          <div className="bg-card rounded-xl">
            <div className="flex items-center gap-3 mb-4">
              <Database className="w-5 h-5 text-primary" />
              <h3 className="font-semibold">Source Dataset</h3>
            </div>
            <select
              value={selectedDatasetId}
              onChange={e => setSelectedDatasetId(e.target.value)}
              className="w-full bg-muted/40 rounded-xl px-4 py-3 text-sm font-medium border-none focus:ring-2 focus:ring-primary/30 appearance-none"
              disabled={isTraining}
            >
              <option value="">Select dataset...</option>
              {datasets.map((ds: Record<string, unknown>) => (
                <option key={String(ds.id)} value={String(ds.id)}>
                  {String(ds.name)} ({String(ds.row_count)} rows)
                </option>
              ))}
            </select>
          </div>

          <div className="border rounded-xl overflow-hidden">
            <div className="p-4 bg-muted/20 border-b">
              <div className="flex gap-6 flex-wrap">
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Target Properties</label>
                  <div className="mt-2 space-y-1.5">
                    {AVAILABLE_TARGETS.map(t => {
                      const ds = datasets.find(d => String(d.id) === String(selectedDatasetId));
                      const cmap = ds?.column_map;
                      let isDisabled = isTraining;
                      
                      if (ds) {
                         if (t.id === "d33" && !ds.has_d33 && (!cmap || !cmap.d33)) isDisabled = true;
                         if (t.id === "tc" && !ds.has_tc && (!cmap || !cmap.tc)) isDisabled = true;
                         if (t.id === "vickers_hardness" && (!cmap || !cmap.vickers_hardness)) isDisabled = true;
                      }

                      return (
                        <label key={t.id} className={`flex items-center gap-2 ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`} title={isDisabled ? "Target not available in selected dataset" : ""}>
                          <input
                            type="checkbox" 
                            className="rounded border-muted-foreground/30 accent-primary"
                            checked={pendingConfig.targets.includes(t.id)}
                            disabled={isDisabled}
                            onChange={() => {
                              const current = pendingConfig.targets;
                              setPendingConfig({
                                ...pendingConfig,
                                targets: current.includes(t.id) ? current.filter(x => x !== t.id) : [...current, t.id]
                              });
                            }}
                          />
                          <span className="text-sm truncate">{t.label}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Algorithm Architecture</label>
                  <select
                    className="mt-2 block w-full bg-muted/40 rounded-lg px-3 py-2 text-sm border-none focus:ring-2 focus:ring-primary/30"
                    value={pendingConfig.model}
                    onChange={(e) => setPendingConfig({ ...pendingConfig, model: e.target.value, params: {} })}
                    disabled={isTraining}
                  >
                    {AVAILABLE_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>
            </div>
            <div className="p-4 bg-muted/10 border-b space-y-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded border-muted-foreground/30 accent-primary"
                  checked={pendingConfig.use_optuna}
                  disabled={isTraining}
                  onChange={(e) => setPendingConfig({ ...pendingConfig, use_optuna: e.target.checked })}
                />
                <span className="text-sm font-semibold text-primary">AI Auto-Tuning (Optuna)</span>
              </label>
              <AnimatePresence>
                {pendingConfig.use_optuna && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                    <div className="flex justify-between items-center mb-1 mt-2">
                       <span className="text-xs text-muted-foreground">Hyperparameter Search Trials</span>
                       <span className="text-xs font-mono">{pendingConfig.optuna_trials}</span>
                    </div>
                    <input
                      type="range"
                      min="10" max="200" step="10"
                      className="w-full accent-primary"
                      value={pendingConfig.optuna_trials}
                      disabled={isTraining}
                      onChange={(e) => setPendingConfig({ ...pendingConfig, optuna_trials: parseInt(e.target.value) })}
                    />
                    <p className="text-[10px] text-muted-foreground mt-2">Optuna will use Bayesian Optimization to search parameter spaces before training. This increases execution time significantly.</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <div className="p-4 bg-card">
              {schema && schema[pendingConfig.model] && schema[pendingConfig.model].params ? (
                <HyperparameterPanel
                  modelName={pendingConfig.model}
                  schema={schema[pendingConfig.model].params}
                  values={pendingConfig.params}
                  disabled={isTraining}
                  onChange={(name, value) => {
                    setPendingConfig(prev => ({ ...prev, params: { ...prev.params, [name]: value } }));
                  }}
                />
              ) : (
                <div className="text-sm text-muted-foreground text-center py-4 animate-pulse">Loading Schema definitions...</div>
              )}
              <div className="mt-4 pt-4 border-t flex justify-end">
                <Button onClick={handleAddPipeline} disabled={isTraining || pendingConfig.targets.length === 0} className="gap-2 w-full sm:w-auto">
                  <Plus className="w-4 h-4" /> Add Pipeline to Queue
                </Button>
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "queue",
      title: "Execution Queue",
      icon: <Network className="w-4 h-4 text-primary" />,
      defaultLayout: { x: 0, y: 7, w: 5, h: 4, minW: 3, minH: 2 },
      component: (
        <div className="flex flex-col h-full bg-card">
          {/* Action Bar */}
          <div className="p-3 border-b flex justify-between items-center bg-muted/10 shrink-0">
             <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Queue Controls</div>
             <div className="flex gap-2">
                {isTraining ? (
                  <Button size="sm" variant="destructive" onClick={stopTraining} className="gap-2 shadow-sm font-semibold">
                    <Square className="w-3.5 h-3.5 fill-current" /> Stop Compute Run
                  </Button>
                ) : (
                  <Button 
                      size="sm"
                      onClick={startTraining} 
                      disabled={jobsQueue.length === 0}
                      className="gap-2 shadow-sm font-semibold bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" /> Initialize Run
                  </Button>
                )}
             </div>
          </div>
          <div className="p-4 flex-1 overflow-auto">
          {jobsQueue.length === 0 ? (
            <div className="border border-dashed rounded-xl p-6 text-center text-muted-foreground bg-muted/20 text-sm">
              No pipelines queued. Initialize one above.
            </div>
          ) : (
            <AnimatePresence>
              <div className="space-y-3">
                {jobsQueue.map((conf, index) => {
                  return (
                    <motion.div 
                      key={conf.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="border rounded-xl bg-card shadow-sm overflow-hidden flex flex-col p-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex gap-3 items-center truncate">
                          <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold shrink-0">
                            {index + 1}
                          </div>
                          <div className="flex flex-col truncate">
                            <span className="font-semibold text-sm">{conf.model} Pipeline {conf.use_optuna ? "⚡ Tuning" : ""}</span>
                            <span className="text-xs text-muted-foreground truncate">
                              {conf.targets.join(", ")} | {conf.use_optuna ? `${conf.optuna_trials} trials` : `${Object.keys(conf.params).length} args`}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <Button disabled={isTraining || index === 0} variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleMoveUp(index)}><ChevronUp className="w-4 h-4" /></Button>
                          <Button disabled={isTraining || index === jobsQueue.length - 1} variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleMoveDown(index)}><ChevronDown className="w-4 h-4" /></Button>
                          <Button disabled={isTraining} variant="ghost" size="icon" className="h-8 w-8 text-blue-500" onClick={() => handleEditPipeline(conf)}><Pencil className="w-4 h-4" /></Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => handleRemovePipeline(conf.id)} disabled={isTraining}><Trash2 className="w-4 h-4" /></Button>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </AnimatePresence>
          )}
        </div>
        </div>
      ),
    },
    {
      key: "terminal",
      title: "Runtime Environment",
      icon: <BrainCircuit className="w-4 h-4 text-primary" />,
      defaultLayout: { x: 5, y: 0, w: 7, h: 7, minW: 4, minH: 4 },
      component: (
        <div className="bg-zinc-950 flex flex-col h-full overflow-hidden">
          <div className="p-3 border-b border-zinc-800 flex justify-between items-center bg-zinc-900/50 shrink-0">
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-primary" />
              <span className="font-mono text-xs font-semibold text-zinc-300">RUNTIME_ENV</span>
            </div>
            {isTraining && (
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span className="text-xs text-emerald-500 font-mono tracking-widest uppercase">Active</span>
              </div>
            )}
          </div>
          <div className="flex-1 overflow-hidden min-h-[300px]">
            <LogTerminal jobId={activeJobLogTarget} onComplete={onJobComplete} onEpochLoss={handleEpochLoss} />
          </div>
        </div>
      ),
    },
    {
      key: "convergence",
      title: "Convergence Metrics (MSE)",
      icon: <Activity className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 0, y: 11, w: 6, h: 4, minW: 4, minH: 3 },
      component: (
        <div className="p-4 h-full">
          <div className="bg-muted/30 rounded-lg p-2 h-[280px]">
            <TrainingLossChart data={graphData} />
          </div>
        </div>
      ),
    },
    {
      key: "comparison",
      title: "Model Comparison Dashboard",
      icon: <Network className="w-4 h-4 text-violet-500" />,
      defaultLayout: { x: 6, y: 11, w: 6, h: 4, minW: 4, minH: 3 },
      component: (
        <div className="p-4 h-full">
          <ModelComparisonDashboard artifacts={artifacts} />
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1400px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Model Studio</h1>
        <p className="text-muted-foreground">Configure pipelines and map them into the execution queue.</p>
      </div>
      <DraggableGrid pageKey="train" cards={cards} />
    </div>
  );
}

