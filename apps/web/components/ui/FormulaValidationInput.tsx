"use client";

/**
 * FormulaValidationInput — Reusable formula input with built-in validation.
 * 
 * Uses the central useFormulaValidation hook.
 * Drop-in for any section needing formula validation (Predict, Optimization, etc.)
 */

import { useEffect } from "react";
import { CheckCircle, XCircle, Loader2, AlertTriangle } from "lucide-react";
import { useFormulaValidation } from "@/lib/hooks/useFormulaValidation";
import { useUIStore } from "@/lib/store/uiStore";

interface FormulaValidationInputProps {
  /** Current formula value (controlled) */
  value: string;
  /** Change handler */
  onChange: (value: string) => void;
  /** Placeholder text */
  placeholder?: string;
  /** CSS class for the wrapper */
  className?: string;
  /** Input id */
  id?: string;
  /** Show validation details (elements, warnings) */
  showDetails?: boolean;
  /** Called when validation completes */
  onValidation?: (result: { is_valid: boolean; error?: string | null }) => void;
}

export default function FormulaValidationInput({
  value,
  onChange,
  placeholder = "e.g. K0.5Na0.5NbO3",
  className = "",
  id,
  showDetails = true,
  onValidation,
}: FormulaValidationInputProps) {
  const { strictFormulaMode } = useUIStore();

  const hook = useFormulaValidation({
    initialValue: value,
    defaultStrictMode: strictFormulaMode,
    onValidation: onValidation ? (r) => onValidation({ is_valid: r.is_valid, error: r.error }) : undefined,
  });

  // Sync external value changes
  useEffect(() => {
    if (value !== hook.value) {
      hook.setValue(value);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // Sync strict mode from global store
  useEffect(() => {
    if (strictFormulaMode !== hook.strictMode) {
      hook.setStrictMode(strictFormulaMode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strictFormulaMode]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
    hook.setValue(e.target.value);
  };

  return (
    <div className={`formula-validation-wrap ${className}`}>
      <div className="formula-input-wrapper">
        <input
          id={id}
          type="text"
          className={hook.inputClassName}
          placeholder={placeholder}
          value={value}
          onChange={handleChange}
          autoComplete="off"
          spellCheck={false}
        />
        <div className="formula-validation-icon">
          {hook.validating ? (
            <Loader2 size={18} className="loading" />
          ) : hook.validation ? (
            hook.validation.is_valid ? (
              <CheckCircle size={18} className="valid" />
            ) : (
              <XCircle size={18} className="invalid" />
            )
          ) : null}
        </div>
      </div>

      {showDetails && hook.validation && hook.validation.is_valid && (
        <div className="formula-details">
          {hook.validation.normalized_formula && (
            <div className="formula-normalized">
              → {hook.validation.normalized_formula}
            </div>
          )}
          {hook.validation.elements && (
            <div className="formula-elements">
              {Object.entries(hook.validation.elements).map(([el, amt]) => (
                <span key={el} className="formula-element-tag">
                  {el}
                  {amt !== 1 ? <sub>{amt}</sub> : null}
                </span>
              ))}
            </div>
          )}
          {hook.validation.warnings.map((w, i) => (
            <div key={i} className="formula-warning">
              <AlertTriangle size={11} /> {w}
            </div>
          ))}
        </div>
      )}

      {showDetails && hook.validation && !hook.validation.is_valid && (
        <div className="formula-error">
          <XCircle size={14} />
          <span>{hook.validation.error || "Invalid formula"}</span>
        </div>
      )}

      {showDetails && hook.validation?.unsupported && hook.validation.unsupported.length > 0 && (
        <div className="formula-unsupported-list">
          <h4>Unsupported: {hook.validation.unsupported.join(", ")}</h4>
          <p style={{ fontSize: 11, color: "var(--text-muted)" }}>
            The model was trained on perovskite piezoelectric elements only.
          </p>
        </div>
      )}
    </div>
  );
}
