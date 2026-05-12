/**
 * Interpret API Client — typed HTTP functions for SHAP, Physics, PySR.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const BASE = `${API_BASE}/api/v1/interpret`;

// ---------- Types ----------

export interface InterpretModel {
  id: string;
  display_name: string;
  target: string;
  algorithm: string;
  r2_score: number;
  rmse: number;
  n_train_samples: number;
  n_test_samples: number;
  feature_dim: number;
  is_default: boolean;
}

export interface BeeswarmFeature {
  name: string;
  mean_abs_shap: number;
  rank: number;
}

export interface ShapBeeswarmResult {
  model_id: string;
  target: string;
  algorithm: string;
  feature_names: string[];
  shap_values: number[][];
  feature_values: number[][];
  base_value: number;
  mean_abs_shap: number[];
  top_features: BeeswarmFeature[];
  n_samples: number;
}

export interface ShapWaterfallResult {
  model_id: string;
  target: string;
  feature_names: string[];
  shap_values: number[];
  feature_values: number[];
  base_value: number;
  prediction: number;
  sample_index: number;
  n_total_samples: number;
}

export interface ShapDependenceResult {
  model_id: string;
  target: string;
  feature_name: string;
  feature_values: number[];
  shap_values: number[];
  interaction_feature: string | null;
  interaction_values: number[];
}

export interface PhysicsCheckItem {
  feature: string;
  expected_effect: string;
  physics_reason: string;
  actual_effect: string;
  aligned: boolean;
  shap_magnitude: number;
  shap_rank: number;
}

export interface PhysicsValidationResult {
  model_id: string;
  target: string;
  alignment_score: number;
  total_checks: number;
  confirmed: number;
  violations: PhysicsCheckItem[];
  confirmed_checks: PhysicsCheckItem[];
  skipped: string[];
}

export interface EquationItem {
  equation_str: string;
  latex: string;
  complexity: number;
  loss: number;
  r2: number;
  readable: string;
}

export interface ParetoPoint {
  complexity: number;
  loss: number;
  r2: number;
}

export interface SymbolicRegressionResult {
  model_id: string;
  target: string;
  equations: EquationItem[];
  best_equation: EquationItem | null;
  pareto_front: ParetoPoint[];
  n_samples: number;
  n_features: number;
  available: boolean;
  error: string | null;
}

// ---------- API Functions ----------

export async function fetchInterpretModels(): Promise<InterpretModel[]> {
  const res = await fetch(`${BASE}/models`, { signal: AbortSignal.timeout(10000) });
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function runShapBeeswarm(
  modelId: string,
  maxSamples = 200,
): Promise<ShapBeeswarmResult> {
  const res = await fetch(`${BASE}/shap/beeswarm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, max_samples: maxSamples }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `SHAP beeswarm failed: ${res.statusText}`);
  }
  return res.json();
}

export async function runShapWaterfall(
  modelId: string,
  sampleIndex = 0,
): Promise<ShapWaterfallResult> {
  const res = await fetch(`${BASE}/shap/waterfall`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, sample_index: sampleIndex }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `SHAP waterfall failed: ${res.statusText}`);
  }
  return res.json();
}

export async function runShapDependence(
  modelId: string,
  featureName: string,
): Promise<ShapDependenceResult> {
  const res = await fetch(`${BASE}/shap/dependence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, feature_name: featureName }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `SHAP dependence failed: ${res.statusText}`);
  }
  return res.json();
}

export async function runPhysicsValidation(
  modelId: string,
): Promise<PhysicsValidationResult> {
  const res = await fetch(`${BASE}/physics-validation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
    signal: AbortSignal.timeout(120000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Physics validation failed: ${res.statusText}`);
  }
  return res.json();
}

export async function runSymbolicRegression(
  modelId: string,
  opts?: { maxComplexity?: number; nIterations?: number; timeoutSeconds?: number },
): Promise<SymbolicRegressionResult> {
  const res = await fetch(`${BASE}/symbolic-regression`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: modelId,
      max_complexity: opts?.maxComplexity ?? 20,
      n_iterations: opts?.nIterations ?? 40,
      timeout_seconds: opts?.timeoutSeconds ?? 120,
    }),
    signal: AbortSignal.timeout(300000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Symbolic regression failed: ${res.statusText}`);
  }
  return res.json();
}

export async function installPySRBackend(): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${BASE}/install-pysr`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `PySR installation failed: ${res.statusText}`);
  }
  return res.json();
}
