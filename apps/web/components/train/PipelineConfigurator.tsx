/**
 * PipelineConfigurator — Dataset selector, field selector, missing value strategies.
 * Features: Select All / Deselect All for targets and input fields.
 */

"use client";

import { useEffect, useState } from "react";
import { Database, ChevronDown, AlertTriangle, Zap, CheckSquare, Square, Layers } from "lucide-react";
import { useTrainingStore } from "@/lib/store/trainingStore";
import { listDatasets, type DatasetSummary } from "@/lib/api/datasets";
import { validateDataset, type FieldIssue } from "@/lib/api/training";

const TARGET_FIELDS = ["d33", "tc", "vickers_hardness"];
const TARGET_LABELS: Record<string, string> = {
  d33: "d₃₃ (pC/N)",
  tc: "Tc (°C)",
  vickers_hardness: "Hardness (HV)",
};
const INPUT_FIELDS = [
  "qm", "kp", "relative_density_pct", "sintering_temp_c",
  "sintering_method", "ceramic_type", "fabrication_method",
  "matrix_type", "filler_wt_pct", "particle_morphology",
  "particle_size_nm", "surface_treatment",
];
const STRATEGY_LABELS: Record<string, string> = {
  knn: "KNN Imputer",
  mean: "Mean",
  median: "Median",
  mode: "Mode",
  drop: "Drop rows",
};

export default function PipelineConfigurator() {
  const {
    selectedDatasetId, selectedFields, targets,
    missingStrategies, validationIssues,
    setDataset, setSelectedFields, setTargets,
    setMissingStrategy, setValidationIssues, setJobPhase,
  } = useTrainingStore();

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);

  // Load datasets
  useEffect(() => {
    listDatasets()
      .then((ds) => setDatasets(ds.filter((d) => d.status === "ready")))
      .catch(() => {});
  }, []);

  // On dataset select — auto-select fields
  const handleDatasetSelect = (id: string) => {
    setDataset(id);
    // Default targets: d33 + tc (user can toggle vickers_hardness if their data has it)
    const defaultTargets = ["d33", "tc"];
    setTargets(defaultTargets);
    // Auto-select formula + only the selected targets (not all target fields)
    setSelectedFields(["formula", ...defaultTargets]);
    setJobPhase("configuring");
  };

  // Toggle field selection
  const toggleField = (field: string) => {
    if (field === "formula") return; // Always selected
    const next = selectedFields.includes(field)
      ? selectedFields.filter((f) => f !== field)
      : [...selectedFields, field];
    setSelectedFields(next);
  };

  // Toggle target selection
  const toggleTarget = (target: string) => {
    const next = targets.includes(target)
      ? targets.filter((t) => t !== target)
      : [...targets, target];
    if (next.length === 0) return; // At least one target required
    setTargets(next);
    // Ensure targets are also in selected fields
    setSelectedFields([...new Set([...selectedFields, ...next])]);
  };

  // Select All / Deselect All targets
  const allTargetsSelected = TARGET_FIELDS.every(t => targets.includes(t));
  const handleSelectAllTargets = () => {
    if (allTargetsSelected) {
      // Keep only the first target (at least one required)
      setTargets([TARGET_FIELDS[0]]);
    } else {
      setTargets([...TARGET_FIELDS]);
      setSelectedFields([...new Set([...selectedFields, ...TARGET_FIELDS])]);
    }
  };

  // Select All / Deselect All input features
  const allInputsSelected = INPUT_FIELDS.every(f => selectedFields.includes(f));
  const handleSelectAllInputs = () => {
    if (allInputsSelected) {
      // Deselect all inputs, keep targets + formula
      setSelectedFields(["formula", ...targets]);
    } else {
      setSelectedFields([...new Set(["formula", ...targets, ...INPUT_FIELDS])]);
    }
  };

  // Run pre-training validation
  const handleValidate = async () => {
    if (!selectedDatasetId) return;
    setValidating(true);
    try {
      const result = await validateDataset(selectedDatasetId, selectedFields, targets);
      setValidationIssues(result.issues);
      // Set default strategies from validation
      for (const issue of result.issues) {
        setMissingStrategy(issue.field, issue.default_strategy);
      }
    } catch (err) {
      console.error("Validation failed:", err);
    } finally {
      setValidating(false);
    }
  };

  // Auto-validate when fields or targets change
  useEffect(() => {
    if (selectedDatasetId && selectedFields.length > 1) {
      handleValidate();
    }
  }, [selectedDatasetId, selectedFields.length, targets.length]);

  return (
    <div className="pipeline-configurator">
      {/* Dataset Selector */}
      <div className="config-section">
        <h3 className="config-section-title">
          <Database size={16} /> Dataset
        </h3>
        <select
          className="config-select"
          value={selectedDatasetId || ""}
          onChange={(e) => handleDatasetSelect(e.target.value)}
        >
          <option value="">Select a dataset...</option>
          {datasets.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.display_name} ({ds.total_rows} rows)
            </option>
          ))}
        </select>
      </div>

      {selectedDatasetId && (
        <>
          {/* Target Selection */}
          <div className="config-section">
            <div className="config-section-header">
              <h3 className="config-section-title">
                <Zap size={16} /> Target Variables
              </h3>
              <button
                className="config-toggle-all-btn"
                onClick={handleSelectAllTargets}
                title={allTargetsSelected ? "Deselect All" : "Select All"}
              >
                {allTargetsSelected ? <CheckSquare size={14} /> : <Square size={14} />}
                {allTargetsSelected ? "Deselect All" : "Select All"}
              </button>
            </div>
            <div className="config-chips">
              {TARGET_FIELDS.map((t) => (
                <button
                  key={t}
                  className={`config-chip ${targets.includes(t) ? "active" : ""}`}
                  onClick={() => toggleTarget(t)}
                >
                  {TARGET_LABELS[t] || t}
                </button>
              ))}
            </div>
          </div>

          {/* Input Field Selection */}
          <div className="config-section">
            <div className="config-section-header">
              <h3 className="config-section-title">
                <Layers size={16} /> Input Features
              </h3>
              <button
                className="config-toggle-all-btn"
                onClick={handleSelectAllInputs}
                title={allInputsSelected ? "Deselect All" : "Select All"}
              >
                {allInputsSelected ? <CheckSquare size={14} /> : <Square size={14} />}
                {allInputsSelected ? "Deselect All" : "Select All"}
              </button>
            </div>
            <div className="config-chips">
              <span className="config-chip active locked">formula</span>
              {INPUT_FIELDS.map((f) => (
                <button
                  key={f}
                  className={`config-chip ${selectedFields.includes(f) ? "active" : ""}`}
                  onClick={() => toggleField(f)}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {/* Missing Value Strategies */}
          {validationIssues.length > 0 && (
            <div className="config-section">
              <h3 className="config-section-title">
                <AlertTriangle size={16} /> Missing Value Handling
              </h3>
              <p className="config-hint">
                {validationIssues.length} field(s) have missing or sentinel values.
                Choose a strategy for each:
              </p>
              <div className="strategy-grid">
                {validationIssues.map((issue) => (
                  <div key={issue.field} className="strategy-row">
                    <div className="strategy-info">
                      <span className="strategy-field">{issue.field}</span>
                      <span className="strategy-count">
                        {issue.count}/{issue.total} missing
                      </span>
                    </div>
                    <select
                      className="strategy-select"
                      value={missingStrategies[issue.field] || issue.default_strategy}
                      onChange={(e) => setMissingStrategy(issue.field, e.target.value)}
                    >
                      {(issue.allowed_strategies?.length > 0
                        ? issue.allowed_strategies
                        : Object.keys(STRATEGY_LABELS)
                      ).map((key) => (
                        <option key={key} value={key}>
                          {STRATEGY_LABELS[key] || key}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
