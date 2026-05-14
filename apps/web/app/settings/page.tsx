"use client";

import { useEffect, useState } from "react";
import { Settings, RefreshCw } from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";
import SystemEnvironment from "@/components/settings/SystemEnvironment";
import ModelLibrary from "@/components/settings/ModelLibrary";
import PendingElements from "@/components/settings/PendingElements";
import AppConfig from "@/components/settings/AppConfig";
import AiManagement from "@/components/settings/AiManagement";
import DangerZone from "@/components/settings/DangerZone";
import GnnConfig from "@/components/settings/GnnConfig";

type Tab = "overview" | "models" | "elements" | "ai" | "config" | "advanced";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "models", label: "Models" },
  { id: "elements", label: "Elements" },
  { id: "ai", label: "AI / LLM" },
  { id: "config", label: "Configuration" },
  { id: "advanced", label: "Advanced" },
];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const { fetchAll, error } = useSettingsStore();
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchAll();
    setTimeout(() => setRefreshing(false), 500);
  };

  return (
    <div className="page-container settings-page">
      <div className="page-header">
        <div className="page-header-icon">
          <Settings size={22} />
        </div>
        <div className="page-header-text">
          <h1>Settings</h1>
          <p>System configuration, model management, AI providers, and platform controls</p>
        </div>
        <button
          className={`settings-refresh-btn main ${refreshing ? "spinning" : ""}`}
          onClick={handleRefresh}
          title="Refresh all settings"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {error && (
        <div className="settings-error-banner">
          <span>⚠️ {error}</span>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="settings-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`settings-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="settings-content">
        {activeTab === "overview" && (
          <>
            <SystemEnvironment />
            <ModelLibrary />
          </>
        )}

        {activeTab === "models" && (
          <ModelLibrary />
        )}

        {activeTab === "elements" && (
          <PendingElements />
        )}

        {activeTab === "ai" && (
          <AiManagement />
        )}

        {activeTab === "config" && (
          <AppConfig />
        )}

        {activeTab === "advanced" && (
          <>
            <GnnConfig />
            <DangerZone />
          </>
        )}
      </div>
    </div>
  );
}
