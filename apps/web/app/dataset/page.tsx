"use client";

/**
 * Piezo.AI — Dataset Page
 * =========================
 * Main page orchestrator for the dataset section.
 *
 * Modes:
 * 1. No active dataset → DatasetList (multi-dataset management)
 * 2. Wizard active → UploadWizard (4-step upload flow)
 * 3. Active ready dataset → DatasetExplorer + Comparison tabs
 */

import { useCallback, useEffect, useState } from "react";
import { Database, GitCompare, Table2 } from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import { getDataset } from "@/lib/api/datasets";
import DatasetList from "@/components/dataset/DatasetList";
import UploadWizard from "@/components/dataset/UploadWizard";
import DatasetExplorer from "@/components/dataset/DatasetExplorer";
import DatasetComparisonView from "@/components/dataset/DatasetComparisonView";

/* ---------- Sub-tabs for ready datasets ---------- */

type ReadyTab = "explorer" | "comparison";

export default function DatasetPage() {
  const {
    activeDatasetId,
    activeDataset,
    wizardActive,
    setActiveDataset,
    setActiveDatasetId,
    resetWizard,
  } = useDatasetStore();

  /* Local tab for ready datasets */
  const [readyTab, setReadyTab] = useState<ReadyTab>("explorer");

  /* Load active dataset detail when id changes */
  useEffect(() => {
    if (!activeDatasetId) {
      setActiveDataset(null);
      return;
    }
    if (activeDataset?.id === activeDatasetId) return;

    getDataset(activeDatasetId)
      .then(setActiveDataset)
      .catch(() => {
        setActiveDatasetId(null);
        setActiveDataset(null);
      });
  }, [activeDatasetId, activeDataset?.id, setActiveDataset, setActiveDatasetId]);

  /* Handle back to list — fully resets wizard + active dataset */
  const handleBackToList = useCallback(() => {
    resetWizard();
    setActiveDatasetId(null);
    setActiveDataset(null);
  }, [resetWizard, setActiveDatasetId, setActiveDataset]);

  /* Mode 1: No active dataset → list */
  if (!activeDatasetId && !wizardActive) {
    return (
      <div className="page-container">
        <div className="page-header">
          <div className="page-header-icon">
            <Database size={22} />
          </div>
          <div className="page-header-text">
            <h1>Dataset Upload &amp; Management</h1>
            <p>Upload, map, clean, review, and explore CSV datasets</p>
          </div>
        </div>
        <DatasetList />
      </div>
    );
  }

  /* Mode 2: Wizard active (upload flow) */
  if (wizardActive) {
    return (
      <div className="page-container">
        <div className="page-header">
          <div className="page-header-icon">
            <Database size={22} />
          </div>
          <div className="page-header-text">
            <h1>Upload Dataset</h1>
            <p>Follow the steps to upload, map, and review your CSV data</p>
          </div>
          <button className="btn-ghost btn-sm" onClick={handleBackToList} style={{ marginLeft: "auto" }}>
            ← Back to Datasets
          </button>
        </div>
        <UploadWizard />
      </div>
    );
  }

  /* Mode 3: Ready dataset → Explorer + Comparison tabs */
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <Database size={22} />
        </div>
        <div className="page-header-text">
          <h1>{activeDataset?.display_name || "Dataset"}</h1>
          <p>
            {activeDataset?.total_rows} rows × {activeDataset?.total_columns} columns
            {activeDataset?.has_composite_fields && " • Composite fields detected"}
            {activeDataset?.status === "pending" && " • Status: pending (needs re-validation)"}
          </p>
        </div>
        <button className="btn-ghost btn-sm" onClick={handleBackToList} style={{ marginLeft: "auto" }}>
          ← Back to Datasets
        </button>
      </div>

      {/* Tab selector */}
      <div className="dataset-page-tabs">
        <button
          className={`dataset-page-tab${readyTab === "explorer" ? " active" : ""}`}
          onClick={() => setReadyTab("explorer")}
        >
          <Table2 size={16} />
          Explorer
        </button>
        <button
          className={`dataset-page-tab${readyTab === "comparison" ? " active" : ""}`}
          onClick={() => setReadyTab("comparison")}
        >
          <GitCompare size={16} />
          Comparison
        </button>
      </div>

      {/* Tab content */}
      {readyTab === "explorer" && <DatasetExplorer />}
      {readyTab === "comparison" && <DatasetComparisonView />}
    </div>
  );
}
