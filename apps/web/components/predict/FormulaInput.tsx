"use client";

/**
 * FormulaInput — Real-time formula validation with green/red indicators.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { CheckCircle, XCircle, Loader2, AlertTriangle } from "lucide-react";
import { validateFormula } from "@/lib/api/predictions";
import { usePredictStore } from "@/lib/store/predictStore";
import { useUIStore } from "@/lib/store/uiStore";

export default function FormulaInput() {
  const {
    formula,
    setFormula,
    formulaValidation,
    setFormulaValidation,
    formulaValidating,
    setFormulaValidating,
  } = usePredictStore();

  const { strictFormulaMode } = useUIStore();

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setFormula(value);
      setFormulaValidation(null);

      if (debounceRef.current) clearTimeout(debounceRef.current);

      if (!value.trim()) {
        setFormulaValidating(false);
        return;
      }

      setFormulaValidating(true);
      debounceRef.current = setTimeout(async () => {
        try {
          const result = await validateFormula(value, strictFormulaMode);
          setFormulaValidation(result);
        } catch {
          setFormulaValidation({
            formula: value,
            is_valid: false,
            normalized_formula: null,
            elements: null,
            unsupported: null,
            error: "Validation service unavailable",
            warnings: [],
          });
        } finally {
          setFormulaValidating(false);
        }
      }, 400);
    },
    [setFormula, setFormulaValidation, setFormulaValidating],
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const inputClass = formula
    ? formulaValidation
      ? formulaValidation.is_valid
        ? "formula-input valid"
        : "formula-input invalid"
      : "formula-input"
    : "formula-input";

  return (
    <div>
      <div className="formula-input-wrapper">
        <input
          id="formula-input"
          type="text"
          className={inputClass}
          placeholder="e.g. K0.5Na0.5NbO3 or (K0.48Na0.48Li0.04)(Nb0.86Ta0.1Sb0.04)O3"
          value={formula}
          onChange={handleChange}
          autoComplete="off"
          spellCheck={false}
        />
        <div className="formula-validation-icon">
          {formulaValidating ? (
            <Loader2 size={18} className="loading" />
          ) : formulaValidation ? (
            formulaValidation.is_valid ? (
              <CheckCircle size={18} className="valid" />
            ) : (
              <XCircle size={18} className="invalid" />
            )
          ) : null}
        </div>
      </div>

      {/* Validation details */}
      {formulaValidation && formulaValidation.is_valid && (
        <div className="formula-details">
          {formulaValidation.normalized_formula && (
            <div className="formula-normalized">
              → {formulaValidation.normalized_formula}
            </div>
          )}
          {formulaValidation.elements && (
            <div className="formula-elements">
              {Object.entries(formulaValidation.elements).map(([el, amt]) => (
                <span key={el} className="formula-element-tag">
                  {el}
                  {amt !== 1 ? <sub>{amt}</sub> : null}
                </span>
              ))}
            </div>
          )}
          {formulaValidation.warnings.map((w, i) => (
            <div key={i} className="formula-warning">
              <AlertTriangle size={11} /> {w}
            </div>
          ))}
        </div>
      )}

      {/* Error details */}
      {formulaValidation && !formulaValidation.is_valid && (
        <div className="formula-error">
          <XCircle size={14} />
          <span>{formulaValidation.error || "Invalid formula"}</span>
        </div>
      )}

      {/* Unsupported elements */}
      {formulaValidation?.unsupported && formulaValidation.unsupported.length > 0 && (
        <div className="formula-unsupported-list">
          <h4>
            Unsupported: {formulaValidation.unsupported.join(", ")}
          </h4>
          <p style={{ fontSize: 11, color: "var(--text-muted)" }}>
            The model was trained on perovskite piezoelectric elements only.
          </p>
        </div>
      )}
    </div>
  );
}
