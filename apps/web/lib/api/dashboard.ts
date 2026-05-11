/**
 * Piezo.AI — Dashboard API Client
 * ==================================
 * HTTP calls for dashboard stats, model management, and report generation.
 */

import { APP_CONFIG } from "@/lib/constants";

const BASE = `${APP_CONFIG.api.baseUrl}/api/v1/dashboard`;

/* ---------- Types ---------- */

export interface SystemStats {
  dataset_count: number;
  dataset_ready_count: number;
  dataset_pending_count: number;
  total_material_rows: number;
  trained_model_count: number;
  prediction_count: number;
  training_job_count: number;
  training_completed_count: number;
  training_failed_count: number;
  db_size_mb: number;
}

export interface DashboardModel {
  id: string;
  display_name: string;
  target: string;
  algorithm: string;
  r2_score: number;
  rmse: number;
  feature_dim: number;
  n_train_samples: number;
  n_test_samples: number;
  training_duration_s: number;
  is_default: boolean;
  dataset_id: string;
  artifact_dir: string;
  model_file_path: string;
  created_at: string;
}

export interface TargetDistribution {
  target: string;
  count: number;
  percentage: number;
}

export interface PredictionHistoryItem {
  id: string;
  member_ids: string[];
  formula: string;
  is_composite: boolean;
  composite_params: Record<string, unknown> | null;
  d33_predicted: number | null;
  tc_predicted: number | null;
  hardness_predicted: number | null;
  prediction_status: string;
  created_at: string;
}

export interface ReportGenerateRequest {
  include_r2_rmse: boolean;
  include_predicted_vs_actual: boolean;
  include_shap_summary: boolean;
  include_ai_insight: boolean;
  include_material_insight: boolean;
  selected_prediction_ids: string[];
  selected_model_ids: string[];
}

export interface ReportGenerateResponse {
  report_id: string;
  filename: string;
  download_url: string;
  generated_at: string;
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

/** Get system-wide stats */
export function getSystemStats() {
  return apiFetch<SystemStats>(`${BASE}/stats`);
}

/** Get target distribution for donut chart */
export function getTargetDistribution() {
  return apiFetch<TargetDistribution[]>(`${BASE}/target-distribution`);
}

/** List all trained models (extended) */
export function listDashboardModels() {
  return apiFetch<DashboardModel[]>(`${BASE}/models`);
}

/** Rename a model (UUID stays constant) */
export function renameDashboardModel(modelId: string, displayName: string) {
  return apiFetch<DashboardModel>(`${BASE}/models/${modelId}/rename`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

/** Set model as default for its target */
export function setDashboardModelDefault(modelId: string) {
  return apiFetch<DashboardModel>(`${BASE}/models/${modelId}/default`, {
    method: "PATCH",
  });
}

/** Delete a single model */
export function deleteDashboardModel(modelId: string) {
  return apiFetch<void>(`${BASE}/models/${modelId}`, { method: "DELETE" });
}

/** Delete multiple models */
export function bulkDeleteDashboardModels(modelIds: string[]) {
  return apiFetch<{ deleted_count: number; errors: string[] }>(
    `${BASE}/models/bulk-delete`,
    { method: "POST", body: JSON.stringify({ model_ids: modelIds }) },
  );
}

/** Get parsed dataset download URL */
export function getParsedDatasetUrl(modelId: string): string {
  return `${BASE}/models/${modelId}/parsed-dataset`;
}

/** Get prediction history */
export function getPredictionHistory(limit: number = 100) {
  return apiFetch<PredictionHistoryItem[]>(
    `${BASE}/predictions/history?limit=${limit}`,
  );
}

/** Delete a single prediction */
export function deletePrediction(predictionId: string) {
  return apiFetch<void>(`${BASE}/predictions/${predictionId}`, {
    method: "DELETE",
  });
}

/** Delete multiple predictions */
export function bulkDeletePredictions(predictionIds: string[]) {
  return apiFetch<{ deleted_count: number; errors: string[] }>(
    `${BASE}/predictions/bulk-delete`,
    { method: "POST", body: JSON.stringify({ prediction_ids: predictionIds }) },
  );
}

/** Generate a PDF report */
export function generateReport(options: ReportGenerateRequest) {
  return apiFetch<ReportGenerateResponse>(`${BASE}/reports/generate`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

/** Get report download URL */
export function getReportDownloadUrl(reportId: string): string {
  return `${BASE}/reports/${reportId}/download`;
}

/** Check LLM configuration status */
export function getLlmStatus() {
  return apiFetch<{ configured: boolean; provider: string; model: string }>(
    `${BASE}/llm-status`,
  );
}

/** Get parsed elemental compositions for comparison view */
export function getParsedCompositions(datasetId: string) {
  return apiFetch<{
    rows: Record<string, unknown>[];
    columns: string[];
    found: boolean;
  }>(`${BASE}/datasets/${datasetId}/parsed-compositions`);
}

/* ---------- Dataset Management (from Dashboard) ---------- */

/** Rename a dataset from dashboard */
export function renameDashboardDataset(datasetId: string, displayName: string) {
  return apiFetch<{ id: string; display_name: string; status: string }>(
    `${BASE}/datasets/${datasetId}/rename`,
    { method: "PATCH", body: JSON.stringify({ display_name: displayName }) },
  );
}

/** Copy a dataset from dashboard */
export function copyDashboardDataset(datasetId: string, newName?: string) {
  return apiFetch<{ id: string; display_name: string; status: string }>(
    `${BASE}/datasets/${datasetId}/copy`,
    { method: "POST", body: JSON.stringify({ new_name: newName }) },
  );
}

/** Delete a dataset from dashboard */
export function deleteDashboardDataset(datasetId: string) {
  return apiFetch<void>(`${BASE}/datasets/${datasetId}`, { method: "DELETE" });
}

/* ---------- Parse Dataset on Demand ---------- */

/** Parse a dataset's formulas into elemental decomposition */
export function parseDataset(datasetId: string, strictMode: boolean = true) {
  return apiFetch<{
    rows: Record<string, unknown>[];
    columns: string[];
    total_parsed: number;
    total_skipped: number;
    skipped_details: { uid: number; reason: string }[];
  }>(`${BASE}/datasets/${datasetId}/parse`, {
    method: "POST",
    body: JSON.stringify({ strict_mode: strictMode }),
  });
}

