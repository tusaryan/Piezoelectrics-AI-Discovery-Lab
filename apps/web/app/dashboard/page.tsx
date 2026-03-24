"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Database, Folder, Activity, Layers, Trash2, RefreshCw, Plus,
  BarChart3, FileText, Zap, TrendingUp, Cpu, Download, ExternalLink,
  FlaskConical, BrainCircuit, Diamond, Beaker
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from "recharts";
import { DraggableGrid, CardDefinition } from "@/components/layout/DraggableGrid";

interface SystemStats {
  total_materials: number;
  total_datasets: number;
  total_models: number;
  total_training_jobs: number;
}

interface DatasetInfo {
  id: string;
  name: string;
  status: string;
  row_count: number | null;
  has_d33: boolean;
  has_tc: boolean;
  created_at: string | null;
}

interface ModelInfo {
  id: string;
  model_name: string;
  target: string;
  algorithm: string;
  r2_score: number | null;
  rmse: number | null;
  is_active: boolean;
  created_at: string;
}

export default function DashboardPage() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, datasetsRes, modelsRes] = await Promise.all([
        fetch("/api/v1/system/stats", { cache: "no-store" }),
        fetch("/api/v1/datasets", { cache: "no-store" }),
        fetch("/api/v1/training/models", { cache: "no-store" }).catch(() => null),
      ]);
      if (statsRes.ok) {
        const statsJson = await statsRes.json();
        if (statsJson.data) setStats(statsJson.data);
      }
      if (datasetsRes.ok) {
        const datasetsJson = await datasetsRes.json();
        if (datasetsJson.data) setDatasets(datasetsJson.data);
      }
      if (modelsRes && modelsRes.ok) {
        const modelsJson = await modelsRes.json();
        if (modelsJson.data) setModels(modelsJson.data);
      }
    } catch (e) {
      console.error("Failed to fetch dashboard data", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selected.size === datasets.length) setSelected(new Set());
    else setSelected(new Set(datasets.map((d) => d.id)));
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete dataset "${name}" and all its materials?`)) return;
    setDeleting(id);
    try {
      const res = await fetch(`/api/v1/datasets/${id}`, { method: "DELETE" });
      if (res.ok) { setSelected((prev) => { const n = new Set(prev); n.delete(id); return n; }); await fetchAll(); }
      else alert("Failed to delete.");
    } catch { alert("Network error."); } finally { setDeleting(null); }
  };

  const handleBulkDelete = async () => {
    if (selected.size === 0) return;
    if (!confirm(`Delete ${selected.size} dataset(s)?`)) return;
    setBulkDeleting(true);
    try {
      await Promise.all(Array.from(selected).map((id) => fetch(`/api/v1/datasets/${id}`, { method: "DELETE" })));
      setSelected(new Set()); await fetchAll();
    } catch { alert("Some deletions may have failed."); } finally { setBulkDeleting(false); }
  };

  const handleGenerateReport = async () => {
    setGeneratingReport(true);
    try {
      const res = await fetch("/api/v1/predict/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: "dashboard-full", include_all: true })
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = "piezo_ai_full_report.pdf";
        document.body.appendChild(a); a.click(); a.remove();
      }
    } catch (e) { console.error(e); }
    setGeneratingReport(false);
  };

  const statCards = [
    { label: "Total Materials", value: stats?.total_materials, icon: Database, color: "text-primary", bg: "bg-primary/10" },
    { label: "Datasets", value: stats?.total_datasets, icon: Folder, color: "text-emerald-500", bg: "bg-emerald-500/10" },
    { label: "Models Trained", value: stats?.total_models, icon: Layers, color: "text-purple-500", bg: "bg-purple-500/10" },
    { label: "Training Jobs", value: stats?.total_training_jobs, icon: Activity, color: "text-amber-500", bg: "bg-amber-500/10" },
  ];

  const statusColor = (status: string) => {
    switch (status) {
      case "ready": return "text-emerald-600 bg-emerald-500/10";
      case "processing": return "text-blue-600 bg-blue-500/10";
      case "error": return "text-red-600 bg-red-500/10";
      case "pending_mapping": return "text-amber-600 bg-amber-500/10";
      default: return "text-muted-foreground bg-muted";
    }
  };

  // Model performance chart data
  const modelChartData = models.map((m) => ({
    name: m.model_name.length > 12 ? m.model_name.slice(0, 12) + "…" : m.model_name,
    r2: m.r2_score != null ? Number(m.r2_score) : 0,
    rmse: m.rmse != null ? Number(m.rmse) : 0,
    target: m.target,
  }));

  // Target distribution pie chart
  const targetCounts = models.reduce((acc, m) => {
    acc[m.target] = (acc[m.target] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  const pieData = Object.entries(targetCounts).map(([name, value]) => ({ name, value }));
  const PIE_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  // Quick link cards
  const quickLinks = [
    { label: "Predict", href: "/predict", icon: Zap, color: "text-indigo-500", bg: "bg-indigo-500/10", desc: "Run predictions" },
    { label: "Train", href: "/train", icon: TrendingUp, color: "text-emerald-500", bg: "bg-emerald-500/10", desc: "Train models" },
    { label: "Inverse Design", href: "/inverse", icon: FlaskConical, color: "text-violet-500", bg: "bg-violet-500/10", desc: "Discover compositions" },
    { label: "Interpretability", href: "/interpret", icon: BrainCircuit, color: "text-amber-500", bg: "bg-amber-500/10", desc: "Explain models" },
    { label: "Composites", href: "/composite", icon: Beaker, color: "text-teal-500", bg: "bg-teal-500/10", desc: "Design composites" },
    { label: "Hardness", href: "/hardness", icon: Diamond, color: "text-blue-500", bg: "bg-blue-500/10", desc: "Predict hardness" },
  ];

  const cards: CardDefinition[] = [
    {
      key: "stats",
      title: "System Overview",
      icon: <Activity className="w-4 h-4 text-primary" />,
      defaultLayout: { x: 0, y: 0, w: 12, h: 2, minW: 6, minH: 2 },
      component: (
        <div className="p-4 grid gap-4 grid-cols-2 lg:grid-cols-4">
          {statCards.map((card, idx) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="rounded-xl border border-border bg-card text-card-foreground p-4 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex flex-row items-center justify-between pb-2">
                <h3 className="tracking-tight text-sm font-medium">{card.label}</h3>
                <div className={`p-2 rounded-full ${card.bg}`}>
                  <card.icon className={`h-4 w-4 ${card.color}`} />
                </div>
              </div>
              <div className="text-3xl font-bold mt-2">
                {loading ? <div className="h-9 w-16 bg-muted animate-pulse rounded" /> : (stats ? card.value : "--")}
              </div>
            </motion.div>
          ))}
        </div>
      ),
    },
    {
      key: "quick-links",
      title: "Quick Actions",
      icon: <ExternalLink className="w-4 h-4 text-indigo-500" />,
      defaultLayout: { x: 0, y: 2, w: 6, h: 3, minW: 4, minH: 2 },
      component: (
        <div className="p-4 grid grid-cols-2 lg:grid-cols-3 gap-3">
          {quickLinks.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              className="rounded-xl border p-3 hover:bg-muted/40 transition-colors group flex flex-col gap-2"
            >
              <div className={`p-2 rounded-lg ${link.bg} w-fit`}>
                <link.icon className={`w-4 h-4 ${link.color}`} />
              </div>
              <div>
                <p className="text-sm font-semibold group-hover:text-primary transition-colors">{link.label}</p>
                <p className="text-[10px] text-muted-foreground">{link.desc}</p>
              </div>
            </Link>
          ))}
        </div>
      ),
    },
    {
      key: "model-performance",
      title: "Model Performance (R² Scores)",
      icon: <BarChart3 className="w-4 h-4 text-purple-500" />,
      defaultLayout: { x: 6, y: 2, w: 6, h: 3, minW: 4, minH: 3 },
      component: (
        <div className="p-4 h-full">
          {models.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="currentColor" className="opacity-10" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "currentColor" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "currentColor" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: "var(--bg-card)", borderColor: "var(--border)", borderRadius: "8px", fontSize: "12px" }}
                  formatter={((v: number) => [v.toFixed(4), "R² Score"]) as any}
                />
                <Bar dataKey="r2" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <BarChart3 className="w-8 h-8 opacity-30 mx-auto mb-2" />
                <p className="text-sm">No models trained yet. <Link href="/train" className="text-primary underline">Train one</Link></p>
              </div>
            </div>
          )}
        </div>
      ),
    },
    {
      key: "model-distribution",
      title: "Model Target Distribution",
      icon: <Cpu className="w-4 h-4 text-amber-500" />,
      defaultLayout: { x: 0, y: 5, w: 4, h: 3, minW: 3, minH: 3 },
      component: (
        <div className="p-4 h-full flex items-center justify-center">
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={40} outerRadius={70} fill="#8884d8" paddingAngle={5} dataKey="value" label={({ name, percent }) => `${name} (${((percent || 0) * 100).toFixed(0)}%)`}>
                  {pieData.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center text-muted-foreground">
              <Layers className="w-8 h-8 opacity-30 mx-auto mb-2" />
              <p className="text-sm">No models to analyze</p>
            </div>
          )}
        </div>
      ),
    },
    {
      key: "report",
      title: "Report Generation",
      icon: <FileText className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 4, y: 5, w: 8, h: 3, minW: 4, minH: 2 },
      component: (
        <div className="p-6">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-emerald-500/10">
              <FileText className="w-8 h-8 text-emerald-500" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-lg mb-1">Comprehensive AI Report</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Generate a detailed PDF report with all model performance metrics, SHAP explanations,
                prediction accuracy plots, Pareto optimization results, and AI-generated insights
                about your piezoelectric material discoveries.
              </p>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
                {[
                  { label: "Model Metrics", desc: "R², RMSE, MAE" },
                  { label: "SHAP Analysis", desc: "Feature importance" },
                  { label: "Pareto Front", desc: "Multi-objective" },
                  { label: "AI Insights", desc: "Generated review" },
                ].map((item) => (
                  <div key={item.label} className="bg-muted/30 rounded-lg p-2.5 text-center">
                    <p className="text-xs font-semibold">{item.label}</p>
                    <p className="text-[10px] text-muted-foreground">{item.desc}</p>
                  </div>
                ))}
              </div>
              <button
                onClick={handleGenerateReport}
                disabled={generatingReport || (!stats?.total_models)}
                className="inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition-colors disabled:opacity-50"
              >
                {generatingReport ? (
                  <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Generating...</>
                ) : (
                  <><Download className="w-4 h-4" /> Generate Full Report</>
                )}
              </button>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "datasets",
      title: "Dataset Management",
      icon: <Database className="w-4 h-4 text-emerald-500" />,
      defaultLayout: { x: 0, y: 8, w: 12, h: 5, minW: 6, minH: 3 },
      component: (
        <div>
          <div className="flex items-center justify-between p-4 border-b">
            <p className="text-sm text-muted-foreground">Manage uploaded datasets. Select multiple to bulk delete.</p>
            <div className="flex items-center gap-2">
              {selected.size > 0 && (
                <button onClick={handleBulkDelete} disabled={bulkDeleting} className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors disabled:opacity-50">
                  <Trash2 className={`w-4 h-4 ${bulkDeleting ? "animate-spin" : ""}`} />
                  {bulkDeleting ? "Deleting..." : `Delete ${selected.size}`}
                </button>
              )}
              <Link href="/dataset" className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
                <Plus className="w-4 h-4" /> Upload
              </Link>
            </div>
          </div>
          {loading ? (
            <div className="p-6 space-y-3">
              {[1, 2, 3].map((i) => (<div key={i} className="h-12 bg-muted animate-pulse rounded-lg" />))}
            </div>
          ) : datasets.length === 0 ? (
            <div className="p-12 text-center text-muted-foreground">
              <Folder className="w-12 h-12 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium">No datasets yet</p>
              <p className="text-sm mt-1">Upload your first dataset to get started.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="p-4 w-10"><input type="checkbox" checked={selected.size === datasets.length && datasets.length > 0} onChange={toggleSelectAll} className="w-4 h-4 rounded border-border accent-primary cursor-pointer" /></th>
                    <th className="text-left p-4 font-medium">Name</th>
                    <th className="text-left p-4 font-medium">Status</th>
                    <th className="text-right p-4 font-medium">Materials</th>
                    <th className="text-center p-4 font-medium">d33</th>
                    <th className="text-center p-4 font-medium">Tc</th>
                    <th className="text-left p-4 font-medium">Created</th>
                    <th className="text-right p-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence>
                    {datasets.map((ds) => (
                      <motion.tr key={ds.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className={`border-b last:border-b-0 transition-colors ${selected.has(ds.id) ? "bg-primary/5" : "hover:bg-muted/20"}`}>
                        <td className="p-4"><input type="checkbox" checked={selected.has(ds.id)} onChange={() => toggleSelect(ds.id)} className="w-4 h-4 rounded border-border accent-primary cursor-pointer" /></td>
                        <td className="p-4 font-medium truncate max-w-[200px]" title={ds.name}>{ds.name}</td>
                        <td className="p-4"><span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${statusColor(ds.status)}`}>{ds.status === "pending_mapping" ? "pending" : ds.status}</span></td>
                        <td className="p-4 text-right tabular-nums">{ds.row_count ?? "-"}</td>
                        <td className="p-4 text-center">{ds.has_d33 ? "✓" : "—"}</td>
                        <td className="p-4 text-center">{ds.has_tc ? "✓" : "—"}</td>
                        <td className="p-4 text-muted-foreground whitespace-nowrap">{ds.created_at ? new Date(ds.created_at).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) : "-"}</td>
                        <td className="p-4 text-right">
                          <button onClick={() => handleDelete(ds.id, ds.name)} disabled={deleting === ds.id} className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-red-600 hover:bg-red-500/10 transition-colors disabled:opacity-50">
                            <Trash2 className={`w-3.5 h-3.5 ${deleting === ds.id ? "animate-spin" : ""}`} />
                            {deleting === ds.id ? "..." : "Delete"}
                          </button>
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="max-w-[1400px] mx-auto p-6 md:p-10">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">Dashboard</h1>
          <p className="text-muted-foreground">System overview, model insights, and dataset management.</p>
        </div>
        <button onClick={fetchAll} className="inline-flex items-center gap-2 px-3 py-2 text-sm rounded-lg border border-border hover:bg-muted transition-colors">
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} /> Refresh
        </button>
      </div>
      <DraggableGrid pageKey="dashboard" cards={cards} />
    </div>
  );
}
