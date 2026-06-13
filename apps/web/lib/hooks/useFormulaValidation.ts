/**
 * useFormulaValidation — Central formula validation hook.
 * 
 * Reusable across the entire app: Predict, Optimization Lab, Dataset, Training, etc.
 * Uses the backend /api/v1/predictions/validate-formula endpoint.
 * 
 * Features:
 * - Debounced validation (configurable delay)
 * - Strict mode toggle support
 * - Loading, error, validation result states
 * - Auto-cleanup on unmount
 * 
 * Usage:
 *   const { value, setValue, validation, validating, strictMode, setStrictMode } = useFormulaValidation();
 *   <input value={value} onChange={(e) => setValue(e.target.value)} />
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { validateFormula, type FormulaValidation } from "@/lib/api/predictions";

interface UseFormulaValidationOptions {
  /** Initial formula value */
  initialValue?: string;
  /** Debounce delay in ms (default: 400) */
  debounceMs?: number;
  /** Use strict mode by default */
  defaultStrictMode?: boolean;
  /** Callback when validation completes */
  onValidation?: (result: FormulaValidation) => void;
}

export function useFormulaValidation(options: UseFormulaValidationOptions = {}) {
  const {
    initialValue = "",
    debounceMs = 400,
    defaultStrictMode = false,
    onValidation,
  } = options;

  const [value, setValueRaw] = useState(initialValue);
  const [validation, setValidation] = useState<FormulaValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [strictMode, setStrictMode] = useState(defaultStrictMode);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setValue = useCallback(
    (newValue: string) => {
      setValueRaw(newValue);
      setValidation(null);

      if (debounceRef.current) clearTimeout(debounceRef.current);

      if (!newValue.trim()) {
        setValidating(false);
        return;
      }

      setValidating(true);
      debounceRef.current = setTimeout(async () => {
        try {
          const result = await validateFormula(newValue, strictMode);
          setValidation(result);
          onValidation?.(result);
        } catch {
          const errorResult: FormulaValidation = {
            formula: newValue,
            is_valid: false,
            normalized_formula: null,
            elements: null,
            unsupported: null,
            error: "Validation service unavailable",
            warnings: [],
          };
          setValidation(errorResult);
          onValidation?.(errorResult);
        } finally {
          setValidating(false);
        }
      }, debounceMs);
    },
    [strictMode, debounceMs, onValidation],
  );

  // Re-validate when strict mode changes
  useEffect(() => {
    if (value.trim()) {
      setValue(value);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strictMode]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const isValid = validation?.is_valid ?? null;
  const inputClassName = value
    ? validation
      ? validation.is_valid
        ? "formula-input valid"
        : "formula-input invalid"
      : "formula-input"
    : "formula-input";

  return {
    value,
    setValue,
    validation,
    validating,
    isValid,
    strictMode,
    setStrictMode,
    inputClassName,
    /** Clear all state */
    reset: useCallback(() => {
      setValueRaw("");
      setValidation(null);
      setValidating(false);
      if (debounceRef.current) clearTimeout(debounceRef.current);
    }, []),
  };
}
