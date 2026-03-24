"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Check, ArrowRight, AlertCircle } from "lucide-react";

interface ColumnSuggestion {
  csv_column: string | null;
  confidence: number;
}

interface SchemaColumn {
  internalField: string;
  label: string;
  required: boolean;
}

const EXPECTED_SCHEMA: SchemaColumn[] = [
  { internalField: "formula", label: "Chemical Formula", required: true },
  { internalField: "d33", label: "d₃₃ (pC/N)", required: false },
  { internalField: "tc", label: "Curie Temp. (°C)", required: false },
  { internalField: "sintering_temp", label: "Sintering Temp. (°C)", required: false },
  { internalField: "family_name", label: "Material Family", required: false },
  { internalField: "field_strength", label: "Field Strength", required: false },
  { internalField: "poling_temp", label: "Poling Temp.", required: false },
  { internalField: "density", label: "Density", required: false },
  { internalField: "dielectric_const", label: "Dielectric Constant", required: false },
  { internalField: "dielectric_loss", label: "Dielectric Loss", required: false },
  { internalField: "mech_quality_factor", label: "Mech. Quality Factor", required: false },
];

interface SchemaMapperProps {
  csvColumns: string[];
  suggestedMapping?: Record<string, ColumnSuggestion>;
  onConfirm: (mapping: Record<string, string>) => void;
}

export function SchemaMapper({ csvColumns, suggestedMapping, onConfirm }: SchemaMapperProps) {
  // Initialize mapping from API suggestions (if available) or basic fuzzy match
  const initialMapping = useMemo(() => {
    const m: Record<string, string> = {};
    const cols = csvColumns || [];
    for (const field of EXPECTED_SCHEMA) {
      const suggestion = suggestedMapping?.[field.internalField];
      if (suggestion?.csv_column && suggestion.confidence > 0) {
        m[field.internalField] = suggestion.csv_column;
      } else {
        // Fallback: basic case-insensitive match
        const match = cols.find(
          (c) => c.toLowerCase() === field.internalField.toLowerCase()
        );
        m[field.internalField] = match || "";
      }
    }
    return m;
  }, [csvColumns, suggestedMapping]);

  const [mapping, setMapping] = useState<Record<string, string>>(initialMapping);

  const isValid = mapping.formula !== "";

  // Only show fields that have a suggestion OR are required
  const relevantFields = EXPECTED_SCHEMA.filter(
    (f) => f.required || mapping[f.internalField] !== ""
  );

  // Extra unmapped CSV columns
  const mappedCsvCols = new Set(Object.values(mapping).filter(Boolean));
  const unmappedCols = (csvColumns || []).filter((c) => !mappedCsvCols.has(c));

  return (
    <div className="w-full max-w-4xl mx-auto rounded-xl border bg-card p-6 shadow-sm">
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-1">Map Dataset Columns</h3>
        <p className="text-sm text-muted-foreground">
          Match your CSV headers to the Piezo.AI internal schema. At minimum, map the{" "}
          <strong>Chemical Formula</strong> column.
        </p>
      </div>

      <div className="space-y-3">
        {relevantFields.map((field, idx) => {
          const suggestion = suggestedMapping?.[field.internalField];
          const confidence = suggestion?.confidence || 0;
          return (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              key={field.internalField}
              className="flex items-center gap-4 p-3 rounded-lg border bg-background/50"
            >
              <div className="w-1/3 flex items-center gap-2">
                <span className="font-mono text-sm font-semibold">{field.label}</span>
                {field.required && <span className="text-destructive text-xs">required</span>}
              </div>

              <ArrowRight className="w-4 h-4 text-muted-foreground/50 shrink-0" />

              <div className="w-1/2">
                <select
                  className="w-full h-9 px-3 rounded-md border border-input bg-transparent text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={mapping[field.internalField]}
                  onChange={(e) =>
                    setMapping({ ...mapping, [field.internalField]: e.target.value })
                  }
                >
                  <option value="">-- Not Mapped --</option>
                  {csvColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              <div className="w-16 flex items-center justify-end gap-1">
                {mapping[field.internalField] ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-500" />
                    {confidence > 0 && (
                      <span className="text-xs text-muted-foreground">
                        {Math.round(confidence * 100)}%
                      </span>
                    )}
                  </>
                ) : field.required ? (
                  <AlertCircle className="w-4 h-4 text-destructive" />
                ) : null}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Show unmapped columns for transparency */}
      {unmappedCols.length > 0 && (
        <div className="mt-4 p-3 rounded-lg bg-muted/50 border border-dashed">
          <p className="text-xs font-medium text-muted-foreground mb-2">
            Unmapped CSV columns (will be ignored):
          </p>
          <div className="flex flex-wrap gap-1.5">
            {unmappedCols.map((col) => (
              <span
                key={col}
                className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5 text-xs font-mono text-muted-foreground"
              >
                {col}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 flex justify-between items-center">
        <p className="text-sm text-muted-foreground">
          {Object.values(mapping).filter(Boolean).length} of {csvColumns.length} columns mapped
        </p>
        <button
          onClick={() => onConfirm(mapping)}
          disabled={!isValid}
          className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-10 px-8"
        >
          Confirm Mapping & Validate
        </button>
      </div>
    </div>
  );
}
