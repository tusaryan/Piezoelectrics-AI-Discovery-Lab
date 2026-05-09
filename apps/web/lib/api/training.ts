/**
 * Piezo.AI — Training API Client
 * ================================
 * HTTP calls for the training pipeline.
 */

import { APP_CONFIG } from "@/lib/constants";

const BASE = `${APP_CONFIG.api.baseUrl}/api/v1/training`;

/* ---------- Types ---------- */

export interface HyperparameterDef {
  type: "int" | "float" | "select";
  min: number | null;
  max: number | null;
  step: number | null;
  default: string | number;
  options: string[] | null;
  description: string;
  impact: string;
  recommended: string | number;
}

export interface AlgorithmInfo {
  key: string;
  display_name: string;
  description: string;
  supports_convergence: boolean;
  hyperparameters: Record<string, HyperparameterDef>;
}

export interface FieldIssue {
  field: string;
  issue_type: string;
  count: number;
  total: number;
  message: string;
  suggestion: string;
  default_strategy: string;
  allowed_strategies: string[];
}

export interface DatasetValidationResult {
  dataset_id: string;
  total_rows: number;
  issues: FieldIssue[];
  default_strategies: Record<string, string>;
}

export interface TrainingJob {
  id: string;
  dataset_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  mode: "manual" | "auto";
  targets: string[];
  algorithms: Record<string, string>;
  hyperparameters: Record<string, Record<string, number>> | null;
  selected_fields: string[];
  progress_pct: number;
  current_stage: string | null;
  initial_rows: number | null;
  initial_columns: number | null;
  final_rows: number | null;
  final_columns: number | null;
  artifact_dir: string | null;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface TrainedModelInfo {
  id: string;
  display_name: string;
  target: string;
  algorithm: string;
  r2_score: number;
  rmse: number;
  hyperparameters: Record<string, unknown>;
  feature_dim: number;
  n_train_samples: number;
  n_test_samples: number;
  model_file_path: string;
  training_duration_s: number;
  is_default: boolean;
  created_at: string;
}

export interface TrainingResults {
  job_id: string;
  status: string;
  models: TrainedModelInfo[];
  convergence_data: Record<string, { iteration: number; metric: number }[]>;
}

export interface TrainingJobCreateRequest {
  dataset_id: string;
  targets: string[];
  algorithms: Record<string, string>;
  hyperparameters: Record<string, Record<string, number>>;
  selected_fields: string[];
  missing_strategies: Record<string, string>;
  mode: "manual" | "auto";
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

export function getAlgorithms() {
  return apiFetch<AlgorithmInfo[]>(`${BASE}/algorithms`);
}

export function validateDataset(datasetId: string, selectedFields: string[]) {
  return apiFetch<DatasetValidationResult>(`${BASE}/validate`, {
    method: "POST",
    body: JSON.stringify({ dataset_id: datasetId, selected_fields: selectedFields }),
  });
}

export function createTrainingJob(config: TrainingJobCreateRequest) {
  return apiFetch<TrainingJob>(`${BASE}/jobs`, {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function listTrainingJobs() {
  return apiFetch<TrainingJob[]>(`${BASE}/jobs`);
}

export function getTrainingJob(jobId: string) {
  return apiFetch<TrainingJob>(`${BASE}/jobs/${jobId}`);
}

export function stopTrainingJob(jobId: string) {
  return apiFetch<{ status: string }>(`${BASE}/jobs/${jobId}/stop`, {
    method: "POST",
  });
}

export function deleteTrainingJob(jobId: string) {
  return apiFetch<void>(`${BASE}/jobs/${jobId}`, { method: "DELETE" });
}

export function getTrainingResults(jobId: string) {
  return apiFetch<TrainingResults>(`${BASE}/jobs/${jobId}/results`);
}
