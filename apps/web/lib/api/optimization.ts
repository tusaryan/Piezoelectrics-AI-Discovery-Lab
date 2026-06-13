/**
 * Optimization API Client — typed HTTP functions for structural analysis & NSGA-II.
 */

const BASE = "/api/v1/optimization";

// ---------- Types ----------

export interface OptimizationModel {
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

// Structural Analysis
export interface StructuralDescriptor {
  formula: string;
  normalized_formula: string;
  is_valid: boolean;
  error: string | null;
  tolerance_factor: number;
  octahedral_factor: number;
  crystal_system: string;
  stability_class: string;
  avg_bond_valence_a: number;
  avg_bond_valence_b: number;
  bond_valence_mismatch: number;
  a_site_elements: Record<string, number>;
  b_site_elements: Record<string, number>;
  dopant_elements: Record<string, number>;
  oxygen_content: number;
  total_elements: number;
  avg_electronegativity: number;
  electronegativity_diff: number;
  avg_atomic_mass: number;
  avg_ionic_radius_a: number;
  avg_ionic_radius_b: number;
  polarizability_index: number;
  a_site_variance: number;
  b_site_variance: number;
  is_perovskite_likely: boolean;
  perovskite_confidence: number;
  phase_count: number;
  warnings: string[];
}

// Optimization
export interface ObjectiveConfig {
  direction: string;
  min: number;
  max: number;
  weight: number;
}

export interface ParetoSolution {
  composition: Record<string, number>;
  formula_approx: string;
  predicted: Record<string, number>;
  use_case_tag: string;
  use_case_color: string;
  rank: number;
  crowding_distance: number;
}

export interface OptimizationResult {
  solutions: ParetoSolution[];
  convergence: Record<string, number>[];
  n_generations_run: number;
  n_evaluations: number;
  duration_seconds: number;
  targets_optimized: string[];
  preset_used: string;
  error: string | null;
}

export interface UseCasePreset {
  key: string;
  label: string;
  description: string;
  objectives: Record<string, ObjectiveConfig>;
}

export interface PresetsResponse {
  presets: UseCasePreset[];
}

// ---------- API Functions ----------

export async function fetchOptimizationModels(): Promise<OptimizationModel[]> {
  const res = await fetch(`${BASE}/models`, {
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function runStructuralAnalysis(
  formula: string,
): Promise<StructuralDescriptor> {
  const res = await fetch(`${BASE}/structural-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ formula }),
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.detail || `Structural analysis failed: ${res.statusText}`,
    );
  }
  return res.json();
}

export async function runStructuralComparison(
  formulas: string[],
): Promise<StructuralDescriptor[]> {
  const res = await fetch(`${BASE}/structural-analysis/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ formulas }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.detail || `Structural comparison failed: ${res.statusText}`,
    );
  }
  return res.json();
}

export interface OptimizationStatusResponse {
  status: "not_started" | "starting" | "loading_models" | "optimizing" | "completed" | "error";
  result?: OptimizationResult;
  error?: string;
  task_key?: string;
  targets?: string[];
  pop_size?: number;
  n_generations?: number;
}

export async function fetchOptimizationStatus(
  taskKey: string,
): Promise<OptimizationStatusResponse> {
  const res = await fetch(`${BASE}/optimize/status/${taskKey}`, {
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) {
    throw new Error(`Failed to check optimization status: ${res.statusText}`);
  }
  return res.json();
}

export async function runOptimization(params: {
  model_ids: Record<string, string>;
  objectives: Record<string, ObjectiveConfig>;
  preset?: string;
  pop_size?: number;
  n_generations?: number;
  seed?: number;
  search_elements?: string[];
}): Promise<OptimizationResult> {
  // Step 1: Start the background task (returns immediately)
  const startRes = await fetch(`${BASE}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_ids: params.model_ids,
      objectives: params.objectives,
      preset: params.preset || "custom",
      pop_size: params.pop_size || 100,
      n_generations: params.n_generations || 50,
      seed: params.seed || 42,
      search_elements: params.search_elements,
    }),
    signal: AbortSignal.timeout(15_000),
  });
  if (!startRes.ok) {
    const err = await startRes.json().catch(() => ({}));
    throw new Error(err.detail || `Optimization failed: ${startRes.statusText}`);
  }

  const startData = await startRes.json();

  // If the result was cached, it comes back as full OptimizationResult
  if (startData.solutions !== undefined) {
    return startData as OptimizationResult;
  }

  // If completed status with result embedded
  if (startData.status === "completed" && startData.result) {
    return startData.result as OptimizationResult;
  }

  // If error
  if (startData.status === "error" || startData.error) {
    return {
      solutions: [],
      convergence: [],
      n_generations_run: 0,
      n_evaluations: 0,
      duration_seconds: 0,
      targets_optimized: [],
      preset_used: params.preset || "custom",
      error: startData.error || "Optimization failed",
    };
  }

  // Step 2: Poll for completion
  const taskKey = startData.task_key;
  if (!taskKey) {
    throw new Error("Backend did not return a task_key for background optimization");
  }

  const POLL_INTERVAL = 3000;
  const MAX_POLL_TIME = 600_000; // 10 minutes max
  const startTime = Date.now();

  while (Date.now() - startTime < MAX_POLL_TIME) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));

    const status = await fetchOptimizationStatus(taskKey);

    if (status.status === "completed" && status.result) {
      return status.result;
    }

    if (status.status === "error") {
      return {
        solutions: [],
        convergence: [],
        n_generations_run: 0,
        n_evaluations: 0,
        duration_seconds: 0,
        targets_optimized: [],
        preset_used: params.preset || "custom",
        error: status.error || "Optimization failed in background",
      };
    }

    // Still computing — continue polling
    if (status.targets) {
      console.info(
        `[Optimization] Computing: ${status.targets.join(", ")} (pop=${status.pop_size}, gen=${status.n_generations})...`
      );
    }
  }

  throw new Error(
    "Optimization timed out after 10 minutes. " +
    "The result may still complete in the background — try refreshing the page."
  );
}

export async function fetchPresets(): Promise<UseCasePreset[]> {
  const res = await fetch(`${BASE}/presets`, {
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) throw new Error(`Failed to fetch presets: ${res.statusText}`);
  const data: PresetsResponse = await res.json();
  return data.presets;
}
