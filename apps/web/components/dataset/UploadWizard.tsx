"use client";

/**
 * Piezo.AI — Upload Wizard Container
 * =====================================
 * 4-step wizard: Upload → Map → Review → Explore
 * With step indicator, animated transitions, and navigation.
 */

import { useMemo } from "react";
import { Upload, Columns3, AlertTriangle, Table2, CheckCircle2 } from "lucide-react";
import { useDatasetStore, type WizardStep } from "@/lib/store/datasetStore";
import UploadStep from "./UploadStep";
import ColumnMappingStep from "./ColumnMappingStep";
import ReviewIssuesStep from "./ReviewIssuesStep";
import DatasetExplorer from "./DatasetExplorer";

/* ---------- Step config ---------- */

interface StepConfig {
  key: WizardStep;
  label: string;
  description: string;
  icon: typeof Upload;
}

const STEPS: StepConfig[] = [
  {
    key: "upload",
    label: "Upload",
    description: "Select and upload your CSV file",
    icon: Upload,
  },
  {
    key: "map",
    label: "Map Columns",
    description: "Map CSV columns to backend fields",
    icon: Columns3,
  },
  {
    key: "review",
    label: "Review Issues",
    description: "Review and fix data quality issues",
    icon: AlertTriangle,
  },
  {
    key: "explore",
    label: "Explore",
    description: "Browse and edit your dataset",
    icon: Table2,
  },
];

/* ---------- Component ---------- */

export default function UploadWizard() {
  const { wizardStep, setWizardStep, activeDatasetId, resetWizard } =
    useDatasetStore();

  const currentIdx = useMemo(
    () => STEPS.findIndex((s) => s.key === wizardStep),
    [wizardStep],
  );

  const currentStep = STEPS[currentIdx];

  /* Can navigate back? */
  const canGoBack = currentIdx > 0;

  const handleBack = () => {
    if (canGoBack) {
      setWizardStep(STEPS[currentIdx - 1].key);
    }
  };

  const handleStartOver = () => {
    resetWizard();
  };

  return (
    <div className="wizard-container">
      {/* Step indicator */}
      <div className="wizard-steps">
        {STEPS.map((step, idx) => {
          const isActive = idx === currentIdx;
          const isCompleted = idx < currentIdx;
          const Icon = step.icon;

          let stepClass = "wizard-step";
          if (isActive) stepClass += " active";
          if (isCompleted) stepClass += " completed";

          return (
            <div key={step.key} className="wizard-step-wrapper">
              <div
                className={stepClass}
                role="button"
                tabIndex={isCompleted ? 0 : -1}
                onClick={() => {
                  if (isCompleted) setWizardStep(step.key);
                }}
                aria-label={`Step ${idx + 1}: ${step.label}`}
              >
                <div className="wizard-step-circle">
                  {isCompleted ? (
                    <CheckCircle2 size={18} />
                  ) : (
                    <Icon size={18} />
                  )}
                </div>
                <span className="wizard-step-label">{step.label}</span>
              </div>
              {idx < STEPS.length - 1 && (
                <div
                  className={`wizard-step-line${isCompleted ? " completed" : ""}`}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Step header */}
      <div className="wizard-step-header">
        <h2>{currentStep.label}</h2>
        <p>{currentStep.description}</p>
      </div>

      {/* Step content */}
      <div className="wizard-content">
        {wizardStep === "upload" && <UploadStep />}
        {wizardStep === "map" && <ColumnMappingStep />}
        {wizardStep === "review" && <ReviewIssuesStep />}
        {wizardStep === "explore" && <DatasetExplorer />}
      </div>

      {/* Bottom navigation */}
      <div className="wizard-nav">
        {canGoBack && wizardStep !== "explore" && (
          <button className="btn-ghost" onClick={handleBack}>
            ← Back
          </button>
        )}
        <div className="wizard-nav-spacer" />
        {wizardStep !== "upload" && (
          <button className="btn-ghost btn-danger-text" onClick={handleStartOver}>
            Start Over
          </button>
        )}
      </div>
    </div>
  );
}
