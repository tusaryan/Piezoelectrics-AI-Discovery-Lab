"use client";

/**
 * Piezo.AI — Column Mapping Step
 * ================================
 * Mandatory mapping of CSV columns to backend field names.
 * Step 2 of the dataset upload wizard.
 *
 * Features:
 * - Categorized backend fields (Required, Core, Processing, Composite, Traceability)
 * - Auto-suggested mappings from backend
 * - Live preview of renamed columns
 * - Formula mapping is mandatory
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Columns3,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  RotateCcw,
  Loader2,
} from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import {
  applyMapping,
  getBackendFields,
  type BackendFieldInfo,
} from "@/lib/api/datasets";

/* ---------- Field category labels ---------- */

const CATEGORY_LABELS: Record<string, string> = {
  required: "Required",
  core: "Core Properties",
  processing: "Processing",
  composite: "Composite",
  traceability: "Traceability",
};

const CATEGORY_ORDER = ["required", "core", "processing", "composite", "traceability"];

/* ---------- Component ---------- */

export default function ColumnMappingStep() {
  const {
    activeDatasetId,
    csvColumns,
    previewRows,
    columnMapping,
    suggestedMapping,
    isMappingSaving,
    mappingError,
    backendFields,
    setColumnMapping,
    updateMapping,
    removeMapping,
    setIsMappingSaving,
    setMappingError,
    setBackendFields,
    setActiveDataset,
    setWizardStep,
  } = useDatasetStore();

  const [localError, setLocalError] = useState<string | null>(null);

  /* Load backend fields on mount */
  useEffect(() => {
    if (backendFields.length === 0) {
      getBackendFields()
        .then(setBackendFields)
        .catch(() => {
          /* Non-critical — use hardcoded fallback */
        });
    }
  }, [backendFields.length, setBackendFields]);

  /* Which backend fields are already mapped */
  const mappedBackendFields = useMemo(
    () => new Set(Object.values(columnMapping)),
    [columnMapping],
  );

  /* Is formula mapped? */
  const formulaMapped = mappedBackendFields.has("formula");

  /* Group backend fields by category */
  const fieldsByCategory = useMemo(() => {
    const groups: Record<string, BackendFieldInfo[]> = {};
    for (const f of backendFields) {
      if (!groups[f.category]) groups[f.category] = [];
      groups[f.category].push(f);
    }
    return groups;
  }, [backendFields]);

  /* Preview mapped data */
  const previewMapped = useMemo(() => {
    if (previewRows.length === 0) return [];
    const reverseMap: Record<string, string> = {};
    for (const [csvCol, backendField] of Object.entries(columnMapping)) {
      reverseMap[backendField] = csvCol;
    }
    return previewRows.slice(0, 5).map((row) => {
      const mapped: Record<string, unknown> = {};
      for (const [backendField, csvCol] of Object.entries(reverseMap)) {
        mapped[backendField] = row[csvCol] ?? "";
      }
      return mapped;
    });
  }, [previewRows, columnMapping]);

  /* Handle mapping change */
  const handleMappingChange = useCallback(
    (csvCol: string, backendField: string) => {
      if (backendField === "__skip__") {
        removeMapping(csvCol);
      } else {
        // If this backend field was already mapped to another CSV column, remove that mapping
        const existingCsvCol = Object.entries(columnMapping).find(
          ([, bf]) => bf === backendField,
        )?.[0];
        if (existingCsvCol && existingCsvCol !== csvCol) {
          removeMapping(existingCsvCol);
        }
        updateMapping(csvCol, backendField);
      }
      setLocalError(null);
    },
    [columnMapping, updateMapping, removeMapping],
  );

  /* Reset to suggestions */
  const handleReset = useCallback(() => {
    setColumnMapping(suggestedMapping);
    setLocalError(null);
  }, [suggestedMapping, setColumnMapping]);

  /* Apply mapping */
  const handleApply = useCallback(async () => {
    if (!formulaMapped) {
      setLocalError("Formula mapping is required — please map a column to 'formula'.");
      return;
    }
    if (!activeDatasetId) return;

    setIsMappingSaving(true);
    setMappingError(null);
    setLocalError(null);

    try {
      const result = await applyMapping(activeDatasetId, columnMapping);
      setActiveDataset(result);
      setIsMappingSaving(false);
      setWizardStep("review");
    } catch (err) {
      setIsMappingSaving(false);
      const msg = err instanceof Error ? err.message : "Failed to apply mapping";
      setMappingError(msg);
      setLocalError(msg);
    }
  }, [
    formulaMapped,
    activeDatasetId,
    columnMapping,
    setIsMappingSaving,
    setMappingError,
    setActiveDataset,
    setWizardStep,
  ]);

  /* Infer data type from preview values */
  const inferType = useCallback(
    (csvCol: string): string => {
      const values = previewRows
        .map((r) => r[csvCol])
        .filter((v) => v != null && v !== "");
      if (values.length === 0) return "text";
      const allNumeric = values.every((v) => !isNaN(Number(v)));
      return allNumeric ? "number" : "text";
    },
    [previewRows],
  );

  const error = localError || mappingError;

  return (
    <div className="mapping-step">
      {/* Header */}
      <div className="mapping-header">
        <div className="mapping-header-info">
          <Columns3 size={18} />
          <span>
            Map your CSV columns to backend fields. <strong>Formula</strong> is required.
          </span>
        </div>
        <button className="btn-ghost" onClick={handleReset}>
          <RotateCcw size={14} />
          Reset Suggestions
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="upload-error">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Mapping grid */}
      <div className="mapping-grid">
        {csvColumns.map((csvCol) => {
          const currentMapping = columnMapping[csvCol] || "";
          const isMapped = !!currentMapping;
          const dtype = inferType(csvCol);
          const sampleValues = previewRows
            .slice(0, 3)
            .map((r) => r[csvCol])
            .filter((v) => v != null && v !== "")
            .map(String);

          return (
            <div
              key={csvCol}
              className={`mapping-row${isMapped ? " mapped" : ""}`}
            >
              {/* Source (CSV column) */}
              <div className="mapping-source">
                <div className="mapping-source-name">{csvCol}</div>
                <div className="mapping-source-meta">
                  <span className={`mapping-type-badge ${dtype}`}>{dtype}</span>
                  {sampleValues.length > 0 && (
                    <span className="mapping-sample" title={sampleValues.join(", ")}>
                      {sampleValues[0]}
                      {sampleValues.length > 1 && `, ${sampleValues[1]}`}
                      {sampleValues.length > 2 && "…"}
                    </span>
                  )}
                </div>
              </div>

              {/* Arrow */}
              <div className="mapping-arrow">
                <ArrowRight size={16} />
              </div>

              {/* Target (backend field select) */}
              <div className="mapping-target">
                <select
                  className={`mapping-select${currentMapping === "formula" ? " required-mapped" : ""}`}
                  value={currentMapping}
                  onChange={(e) => handleMappingChange(csvCol, e.target.value)}
                >
                  <option value="__skip__">— Skip this column —</option>
                  {CATEGORY_ORDER.map((cat) => {
                    const fields = fieldsByCategory[cat];
                    if (!fields) return null;
                    return (
                      <optgroup key={cat} label={CATEGORY_LABELS[cat]}>
                        {fields.map((f) => (
                          <option
                            key={f.name}
                            value={f.name}
                            disabled={
                              mappedBackendFields.has(f.name) &&
                              columnMapping[csvCol] !== f.name
                            }
                          >
                            {f.label}{f.required ? " ★" : ""}
                          </option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
                {isMapped && (
                  <CheckCircle2 size={16} className="mapping-check" />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Preview */}
      {previewMapped.length > 0 && Object.keys(columnMapping).length > 0 && (
        <div className="mapping-preview">
          <h4>Preview (first 5 rows with renamed columns)</h4>
          <div className="mapping-preview-table-wrapper">
            <table className="mapping-preview-table">
              <thead>
                <tr>
                  {Object.values(columnMapping).map((bf) => (
                    <th key={bf}>{bf}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewMapped.map((row, idx) => (
                  <tr key={idx}>
                    {Object.values(columnMapping).map((bf) => (
                      <td key={bf}>{row[bf] == null ? "—" : String(row[bf])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Formula warning */}
      {!formulaMapped && (
        <div className="mapping-warning">
          <AlertCircle size={16} />
          <span>
            <strong>Formula</strong> must be mapped to continue. Select which CSV column contains
            chemical composition formulas.
          </span>
        </div>
      )}

      {/* Apply button */}
      <div className="mapping-actions">
        <button
          className="btn-primary"
          onClick={handleApply}
          disabled={!formulaMapped || isMappingSaving}
        >
          {isMappingSaving ? (
            <>
              <Loader2 size={16} className="spin" />
              Saving...
            </>
          ) : (
            <>
              <CheckCircle2 size={16} />
              Apply Mapping &amp; Continue
            </>
          )}
        </button>
      </div>
    </div>
  );
}
