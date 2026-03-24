"use client";

import { useEffect, useState } from "react";
import { BetaWarning } from "@/components/layout/BetaWarning";
import { Button } from "@/components/ui/button";
import { Trash2, CheckCircle2, Box, Edit2 } from "lucide-react";

export default function ModelsPage() {
  const [models, setModels] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const res = await fetch("/api/v1/training/models");
      const data = await res.json();
      if (data.data) {
        setModels(data.data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this model?")) return;
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
      if (res.ok) {
        alert(`${target.toUpperCase()} Model Activated Successfully!`);
      } else {
        alert("Failed to activate model");
      }
    } catch (e) {
      console.error("Activate failed", e);
    }
  };

  const handleRename = async (id: string, currentAlias: string) => {
    const newAlias = prompt("Enter new model name:", currentAlias);
    if (!newAlias || newAlias === currentAlias) return;
    
    try {
      const res = await fetch(`/api/v1/training/models/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alias: newAlias })
      });
      if (res.ok) {
        setModels(prev => prev.map(m => m.id === id ? { ...m, alias: newAlias } : m));
      } else {
        alert("Failed to rename model");
      }
    } catch (e) {
      console.error("Rename failed", e);
    }
  };

  return (
    <div className="p-6 md:p-10 max-w-6xl mx-auto space-y-8 min-h-[calc(100vh-theme(spacing.16))] flex flex-col">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">Model Registry</h1>
        <p className="text-muted-foreground">
          Manage trained AI models, rename configurations, and set active deployment candidates.
        </p>
      </div>

      <BetaWarning />

      <div className="bg-card border rounded-2xl shadow-sm overflow-hidden flex-1">
        <div className="p-6 border-b bg-muted/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Box className="w-5 h-5 text-primary" />
            <h2 className="font-semibold text-lg">Trained Models Library</h2>
          </div>
          <Button variant="outline" size="sm" onClick={fetchModels}>
            Refresh
          </Button>
        </div>
        
        <div className="p-0 overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted-foreground uppercase bg-muted/10 border-b">
              <tr>
                <th className="px-6 py-4 font-medium">Target</th>
                <th className="px-6 py-4 font-medium">Model Name (Alias)</th>
                <th className="px-6 py-4 font-medium">Algorithm</th>
                <th className="px-6 py-4 font-medium">Test R²</th>
                <th className="px-6 py-4 font-medium">Test RMSE</th>
                <th className="px-6 py-4 font-medium">Date Trained</th>
                <th className="px-6 py-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-muted-foreground">Loading models...</td>
                </tr>
              ) : models.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-muted-foreground">No trained models found</td>
                </tr>
              ) : (
                models.map(m => (
                  <tr key={m.id} className="hover:bg-muted/5 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="bg-primary/10 text-primary px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider border border-primary/20">
                        {m.target}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-medium">
                      <div className="flex items-center gap-2">
                        {m.alias || m.model_name}
                        <button 
                          onClick={() => handleRename(m.id, m.alias || m.model_name)}
                          className="text-muted-foreground hover:text-primary transition-colors"
                          title="Rename Model"
                        >
                          <Edit2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-muted-foreground">{m.model_name}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{m.r2_test?.toFixed(3) || "N/A"}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{m.rmse_test?.toFixed(3) || "N/A"}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-muted-foreground">
                      {new Date(m.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Button 
                          variant="secondary" 
                          size="sm" 
                          className="h-8 gap-1"
                          onClick={() => handleActivate(m.id, m.target)}
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Set Active</span>
                        </Button>
                        <Button 
                          variant="destructive" 
                          size="sm" 
                          className="h-8 px-2"
                          onClick={() => handleDelete(m.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
