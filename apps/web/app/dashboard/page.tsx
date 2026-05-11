"use client";

import { useEffect } from "react";
import { BarChart3, RefreshCw } from "lucide-react";
import { useDashboardStore } from "@/lib/store/dashboardStore";
import StatsCards from "@/components/dashboard/StatsCards";
import QuickActions from "@/components/dashboard/QuickActions";
import DatasetList from "@/components/dashboard/DatasetList";
import ModelLibrary from "@/components/dashboard/ModelLibrary";
import DefaultModelSelector from "@/components/dashboard/DefaultModelSelector";
import TargetDistributionChart from "@/components/dashboard/TargetDistributionChart";
import ReportGenerator from "@/components/dashboard/ReportGenerator";
import "./dashboard.css";

export default function DashboardPage() {
  const {
    stats,
    models,
    targetDistribution,
    predictionHistory,
    loading,
    error,
    fetchAll,
    clearError,
  } = useDashboardStore();

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  return (
    <div className="page-container dashboard-page">
      {/* Header */}
      <div className="page-header">
        <div className="page-header-icon">
          <BarChart3 size={22} />
        </div>
        <div className="page-header-text">
          <h1>Dashboard</h1>
          <p>System overview, quick actions, report generation, model management</p>
        </div>
        <button
          className="refresh-btn"
          onClick={fetchAll}
          disabled={loading}
          title="Refresh dashboard data"
        >
          <RefreshCw size={16} className={loading ? "spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="dashboard-error">
          <span>{error}</span>
          <button onClick={clearError}>×</button>
        </div>
      )}

      {/* Stats Cards */}
      <StatsCards stats={stats} loading={loading} />

      {/* Quick Actions */}
      <QuickActions />

      {/* Two-column layout: Datasets + Target Distribution */}
      <div className="dashboard-two-col">
        <div className="dashboard-col-main">
          <DatasetList />
        </div>
        <div className="dashboard-col-side">
          <TargetDistributionChart data={targetDistribution} />
          <DefaultModelSelector models={models} />
        </div>
      </div>

      {/* Model Library */}
      <ModelLibrary models={models} />

      {/* Report Generation */}
      <ReportGenerator
        predictionHistory={predictionHistory}
        models={models.map((m) => ({
          id: m.id,
          display_name: m.display_name,
          target: m.target,
        }))}
      />
    </div>
  );
}
