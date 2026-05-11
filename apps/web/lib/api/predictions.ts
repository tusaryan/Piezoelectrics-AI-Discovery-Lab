/**
 * Piezo.AI — Predictions API Client
 * ====================================
 * HTTP calls for the prediction pipeline.
 */

import { APP_CONFIG } from "@/lib/constants";

const BASE = `${APP_CONFIG.api.baseUrl}/api/v1/predictions`;

/* ---------- Types ---------- */

export interface PropertyPrediction {
  value: number | null;
  ci_lower: number | null;
  ci_upper: number | null;
}

export interface UseCaseInfo {
  name: string;
  category: string;
  confidence: number;
  description: string;
  icon: string;
  color: string;
}

export interface PredictResponse {
  formula: string;
  is_composite: boolean;
  status: "success" | "unsupported_elements" | "parse_error";
  notes: string | null;
  d33: PropertyPrediction | null;
  tc: PropertyPrediction | null;
  hardness: PropertyPrediction | null;
  use_case: UseCaseInfo | null;
  composite_params: Record<string, unknown> | null;
}

export interface CompositeParams {
  matrix_type?: string;
  filler_wt_pct?: number;
  particle_morphology?: string;
  particle_size_nm?: number;
  surface_treatment?: string;
  fabrication_method?: string;
  sintering_temp_c?: number;
  relative_density_pct?: number;
}

export interface TrainedModelItem {
  id: string;
  display_name: string;
  target: string;
  algorithm: string;
  r2_score: number;
  rmse: number;
  feature_dim: number;
  n_train_samples: number;
  n_test_samples: number;
  supported_elements: string[];
  is_default: boolean;
  created_at: string;
}

export interface FormulaValidation {
  formula: string;
  is_valid: boolean;
  normalized_formula: string | null;
  elements: Record<string, number> | null;
  unsupported: string[] | null;
  error: string | null;
  warnings: string[];
}

export interface BatchPredictSummary {
  batch_id: string;
  total_rows: number;
  success_count: number;
  error_count: number;
  result_file_path: string | null;
  source_filename: string;
}

export interface SupportedElements {
  elements: string[];
  count: number;
}

/* ---------- Helper ---------- */

async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      const json = JSON.parse(body);
      detail = json.detail || json.message || body;
    } catch {
      /* raw text */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

/* ---------- Endpoints ---------- */

/** List all trained models for model selector */
export function listModels() {
  return apiFetch<TrainedModelItem[]>(`${BASE}/models`);
}

/** Rename a trained model */
export function renameModel(modelId: string, displayName: string) {
  return apiFetch<TrainedModelItem>(`${BASE}/models/${modelId}/rename`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

/** Set model as default for its target */
export function setDefaultModel(modelId: string) {
  return apiFetch<TrainedModelItem>(`${BASE}/models/${modelId}/default`, {
    method: "PATCH",
  });
}

/** Run a single prediction */
export function predictSingle(
  formula: string,
  modelId: string,
  compositeParams?: CompositeParams,
) {
  return apiFetch<PredictResponse>(`${BASE}/predict`, {
    method: "POST",
    body: JSON.stringify({
      formula,
      model_id: modelId,
      composite_params: compositeParams || null,
    }),
  });
}

/** Validate formula (real-time) */
export function validateFormula(formula: string, strictMode: boolean = false) {
  return apiFetch<FormulaValidation>(`${BASE}/validate-formula`, {
    method: "POST",
    body: JSON.stringify({ formula, strict_mode: strictMode }),
  });
}

/** Upload CSV for batch prediction */
export async function predictBatchCSV(
  modelId: string,
  file: File,
): Promise<BatchPredictSummary> {
  const formData = new FormData();
  formData.append("model_id", modelId);
  formData.append("file", file);

  const res = await fetch(`${BASE}/predict/batch`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      const json = JSON.parse(body);
      detail = json.detail || json.message || body;
    } catch {
      /* raw text */
    }
    throw new Error(detail);
  }
  return res.json();
}

/** Batch predict from existing dataset */
export function predictBatchFromDataset(modelId: string, datasetId: string) {
  return apiFetch<BatchPredictSummary>(`${BASE}/predict/batch-from-dataset`, {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, dataset_id: datasetId }),
  });
}

/** Get supported elements */
export function getSupportedElements() {
  return apiFetch<SupportedElements>(`${BASE}/supported-elements`);
}

/** Download batch result CSV */
export function getBatchDownloadUrl(batchId: string): string {
  return `${BASE}/batch/${batchId}/download`;
}
