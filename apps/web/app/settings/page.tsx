"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Settings as SettingsIcon, Cpu, Trash2, CheckCircle2, RefreshCw,
  Database, Activity, Pencil, X, Check, Globe, AlertTriangle,
  Server, HardDrive
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";
import { DatasetLibrary } from "@/components/settings/DatasetLibrary";

interface ModelArtifact {
  id: string;
  model_name: string;
  target: string;
  algorithm: string;
  r2_score: number | null;
  rmse: number | null;
  created_at: string;
}

interface SystemStats {
  total_datasets: number;
  total_predictions: number;
  total_models: number;
  db_size_mb: number;
}

// ── Trained Models Library ─────────────────────────────────────
function ModelsLibrary() {
  const [models, setModels] = useState<ModelArtifact[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");

  const fetchModels = useCallback(async () => {
    try {
      setIsLoading(true);
      const res = await fetch("/api/v1/training/models");
      const data = await res.json();
      if (data.data) setModels(data.data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { fetchModels(); }, [fetchModels]);

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this model artifact?")) return;
    try {
      await fetch(`/api/v1/training/models/${id}`, { method: "DELETE" });
      setModels(prev => prev.filter(m => m.id !== id));
    } catch (e) {
      console.error("Failed to delete", e);
    }
  };

  const handleActivate = async (id: string, target: string) => {
    try {
      const res = await fetch(`/api/v1/training/models/${id}/activate`, { method: "POST" });
      if (res.ok) alert(`${target.toUpperCase()} model activated!`);
      else alert("Failed to activate model");
    } catch (e) {
      console.error(e);
    }
  };

  const handleRename = async (id: string) => {
    try {
      await fetch(`/api/v1/training/models/${id}/rename`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: editName }),
      });
      setModels(prev => prev.map(m => m.id === id ? { ...m, model_name: editName } : m));
      setEditingId(null);
    } catch (e) {
      console.error(e);
      setEditingId(null);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex justify-between items-center">
        <p className="text-sm text-muted-foreground">
          Manage all trained model artifacts. Activate, rename, or delete.
        </p>
        <Button onClick={fetchModels} variant="outline" size="sm" className="gap-1.5">
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </Button>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-2 border-t-primary border-primary/20 rounded-full animate-spin" />
        </div>
      ) : models.length === 0 ? (
        <div className="border border-dashed rounded-xl p-8 text-center text-muted-foreground bg-muted/20">
          No models trained yet. Go to the Train page to create your first model.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase text-muted-foreground tracking-wider">
                <th className="pb-3 pr-4">Target</th>
                <th className="pb-3 pr-4">Model Name</th>
                <th className="pb-3 pr-4">Algorithm</th>
                <th className="pb-3 pr-4">Test R²</th>
                <th className="pb-3 pr-4">Test RMSE</th>
                <th className="pb-3 pr-4">Date</th>
                <th className="pb-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} className="border-b last:border-0 hover:bg-muted/20 transition-colors">
                  <td className="py-3 pr-4">
                    <span className="px-2 py-1 text-xs font-semibold rounded-md bg-primary/10 text-primary uppercase">
                      {m.target}
                    </span>
                  </td>
                  <td className="py-3 pr-4">
                    {editingId === m.id ? (
                      <div className="flex items-center gap-1">
                        <input
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="bg-muted/40 border rounded px-2 py-1 text-sm w-32"
                          autoFocus
                          onKeyDown={(e) => e.key === "Enter" && handleRename(m.id)}
                        />
                        <button onClick={() => handleRename(m.id)} className="p-1 hover:bg-muted rounded">
                          <Check className="w-3.5 h-3.5 text-emerald-500" />
                        </button>
                        <button onClick={() => setEditingId(null)} className="p-1 hover:bg-muted rounded">
                          <X className="w-3.5 h-3.5 text-muted-foreground" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5">
                        <span>{m.model_name}</span>
                        <button
                          onClick={() => { setEditingId(m.id); setEditName(m.model_name); }}
                          className="p-0.5 hover:bg-muted rounded opacity-50 hover:opacity-100 transition-opacity"
                        >
                          <Pencil className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </td>
                  <td className="py-3 pr-4 text-muted-foreground">{m.algorithm}</td>
                  <td className="py-3 pr-4 font-mono">
                    {m.r2_score != null ? Number(m.r2_score).toFixed(4) : "N/A"}
                  </td>
                  <td className="py-3 pr-4 font-mono">
                    {m.rmse != null ? Number(m.rmse).toFixed(4) : "N/A"}
                  </td>
                  <td className="py-3 pr-4 text-muted-foreground text-xs">
                    {new Date(m.created_at).toLocaleDateString("en-GB", {
                      day: "2-digit", month: "2-digit", year: "numeric",
                    })}{", "}
                    {new Date(m.created_at).toLocaleTimeString("en-GB", {
                      hour: "2-digit", minute: "2-digit",
                    })}
                  </td>
                  <td className="py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        onClick={() => handleActivate(m.id, m.target)}
                        variant="outline" size="sm"
                        className="gap-1 text-xs h-7"
                      >
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        Set Active
                      </Button>
                      <Button
                        onClick={() => handleDelete(m.id)}
                        variant="ghost" size="icon"
                        className="h-7 w-7 text-destructive hover:bg-destructive/10"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Environment Info ───────────────────────────────────────────
function EnvironmentInfo() {
  const [stats, setStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    fetch("/api/v1/system/stats")
      .then(r => r.json())
      .then(d => { if (d.data) setStats(d.data); })
      .catch(console.error);
  }, []);

  const items = stats
    ? [
        { label: "Datasets", value: stats.total_datasets, icon: <Database className="w-4 h-4 text-blue-500" /> },
        { label: "Predictions", value: stats.total_predictions, icon: <Activity className="w-4 h-4 text-emerald-500" /> },
        { label: "Trained Models", value: stats.total_models, icon: <Cpu className="w-4 h-4 text-violet-500" /> },
        { label: "DB Size", value: `${(stats.db_size_mb || 0).toFixed(1)} MB`, icon: <HardDrive className="w-4 h-4 text-amber-500" /> },
      ]
    : [];

  return (
    <div className="p-4 grid grid-cols-2 gap-4">
      {stats ? (
        items.map((item) => (
          <div key={item.label} className="rounded-lg bg-muted/30 p-4 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-card">{item.icon}</div>
            <div>
              <p className="text-xs text-muted-foreground">{item.label}</p>
              <p className="text-lg font-bold">{item.value}</p>
            </div>
          </div>
        ))
      ) : (
        <div className="col-span-2 py-8 text-center text-muted-foreground animate-pulse">
          Loading system info...
        </div>
      )}
    </div>
  );
}

// ── API Configuration ──────────────────────────────────────────
function APIConfiguration() {
  return (
    <div className="p-4 space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="rounded-lg border p-3 space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wider">Backend URL</p>
          <p className="font-mono text-sm">http://localhost:8000</p>
        </div>
        <div className="rounded-lg border p-3 space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wider">WebSocket</p>
          <p className="font-mono text-sm">ws://localhost:8000/ws</p>
        </div>
        <div className="rounded-lg border p-3 space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wider">Training Engine</p>
          <p className="font-mono text-sm">FastAPI BackgroundTasks</p>
        </div>
        <div className="rounded-lg border p-3 space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wider">Database</p>
          <p className="font-mono text-sm">postgresql://localhost:5432/piezo_ai</p>
        </div>
      </div>
    </div>
  );
}

// ── Danger Zone ────────────────────────────────────────────────
function DangerZone() {
  const handlePurgeModels = async () => {
    if (!confirm("This will delete ALL trained model artifacts. Continue?")) return;
    alert("Purge models (not yet implemented)");
  };

  const handleClearCache = async () => {
    if (!confirm("This will clear all cached predictions and reports. Continue?")) return;
    alert("Cache cleared (not yet implemented)");
  };

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center justify-between p-3 rounded-lg border border-destructive/30 bg-destructive/5">
        <div>
          <p className="text-sm font-medium">Purge All Models</p>
          <p className="text-xs text-muted-foreground">Delete every trained model artifact from the database and filesystem.</p>
        </div>
        <Button variant="destructive" size="sm" onClick={handlePurgeModels} className="shrink-0">
          <Trash2 className="w-3.5 h-3.5 mr-1.5" /> Purge
        </Button>
      </div>
      <div className="flex items-center justify-between p-3 rounded-lg border border-amber-500/30 bg-amber-500/5">
        <div>
          <p className="text-sm font-medium">Clear Prediction Cache</p>
          <p className="text-xs text-muted-foreground">Remove all cached inference results and generated reports.</p>
        </div>
        <Button variant="outline" size="sm" onClick={handleClearCache} className="shrink-0 border-amber-500/40 text-amber-600 hover:bg-amber-500/10">
          Clear
        </Button>
      </div>
    </div>
  );
}

// ── Settings Page ──────────────────────────────────────────────
export default function SettingsPage() {
  const cards: CardDefinition[] = [
    {
      key: "models-library",
      title: "Trained Models Library",
      icon: <Cpu className="w-4 h-4 text-violet-500" />,
      defaultLayout: { x: 0, y: 0, w: 12, h: 5, minW: 6, minH: 3 },
      component: <ModelsLibrary />,
    },
    {
      key: "dataset-library",
      title: "Dataset Library",
      icon: <Database className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 0, y: 5, w: 12, h: 6, minW: 6, minH: 4 },
      component: (
        <div className="h-full overflow-y-auto hide-scrollbar">
           <DatasetLibrary />
        </div>
      ),
    },
    {
      key: "environment",
      title: "System Environment",
      icon: <Server className="w-4 h-4 text-blue-500" />,
      defaultLayout: { x: 0, y: 11, w: 6, h: 3, minW: 4, minH: 2 },
      component: <EnvironmentInfo />,
    },
    {
      key: "api-config",
      title: "API Configuration",
      icon: <Globe className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 6, y: 11, w: 6, h: 3, minW: 4, minH: 2 },
      component: <APIConfiguration />,
    },
    {
      key: "danger-zone",
      title: "Danger Zone",
      icon: <AlertTriangle className="w-4 h-4 text-destructive" />,
      defaultLayout: { x: 0, y: 14, w: 12, h: 3, minW: 6, minH: 2 },
      component: <DangerZone />,
    },
  ];

  return (
    <div className="p-6 md:p-10 max-w-[1400px] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-3">
          <SettingsIcon className="w-8 h-8 text-muted-foreground" />
          System Settings
        </h1>
        <p className="text-muted-foreground">
          Manage trained models, infrastructure configuration, and system maintenance.
        </p>
      </div>

      <DraggableGrid pageKey="settings" cards={cards} />
    </div>
  );
}
