/**
 * Piezo.AI — Settings API Client
 * ================================
 * HTTP client for all settings endpoints.
 */

import { APP_CONFIG } from "../constants";

const API = APP_CONFIG.api.baseUrl;

// ── Helpers ──────────────────────

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || res.statusText);
  }
  return res.json();
}

// ── System Environment ──────────────────────

export interface SystemEnvironment {
  app_name: string;
  app_version: string;
  dataset_count: number;
  total_rows: number;
  trained_model_count: number;
  prediction_count: number;
  db_size_mb: number;
  python_version: string;
  enable_gnn: boolean;
  enable_composite: boolean;
  enable_hardness: boolean;
}

export async function getSystemEnvironment(): Promise<SystemEnvironment> {
  return fetchJson(`${API}/api/v1/settings/system`);
}

// ── App Config ──────────────────────

export type AppConfig = Record<string, string>;

export async function getAppConfig(): Promise<AppConfig> {
  return fetchJson(`${API}/api/v1/settings/config`);
}

export async function updateAppConfig(updates: Record<string, string>): Promise<AppConfig> {
  return fetchJson(`${API}/api/v1/settings/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  });
}

export interface EnvImportResult {
  success: boolean;
  message: string;
  keys_updated: number;
  keys_added: number;
  keys_skipped: number;
  skipped_keys: string[];
}

export async function importEnvFile(file: File): Promise<EnvImportResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API}/api/v1/settings/config/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

export async function uploadLogo(file: File): Promise<{ message: string; path: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API}/api/v1/settings/config/logo`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// ── LLM Config ──────────────────────

export interface LlmConfig {
  provider: string;
  model: string;
  has_api_key: boolean;
  base_url: string;
  temperature: number;
  max_tokens: number;
  status: string;
  status_message: string;
}

export interface LlmProvider {
  id: string;
  name: string;
  description: string;
  requires_api_key: boolean;
  requires_base_url: boolean;
  default_models: string[];
  icon: string;
}

export async function getLlmStatus(): Promise<LlmConfig> {
  return fetchJson(`${API}/api/v1/settings/llm/status`);
}

export async function updateLlmConfig(data: Partial<{
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
}>): Promise<LlmConfig> {
  return fetchJson(`${API}/api/v1/settings/llm`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function getLlmProviders(): Promise<LlmProvider[]> {
  return fetchJson(`${API}/api/v1/settings/llm/providers`);
}

// ── Element Registry ──────────────────────

export interface ElementInfo {
  symbol: string;
  atomic_number: number;
  category: string;
  perovskite_site: string;
  is_rare_earth: boolean;
  is_pending: boolean;
  is_user_added: boolean;
  property_count: number;
}

export interface ElementRegistry {
  supported_elements: ElementInfo[];
  pending_elements: string[];
  user_added_elements: string[];
  total_properties: number;
  property_keys: string[];
  default_property_keys: string[];
  user_added_properties: string[];
}

export async function getElementRegistry(): Promise<ElementRegistry> {
  return fetchJson(`${API}/api/v1/settings/elements`);
}

export async function addPendingElement(
  symbol: string, categories: string[]
): Promise<{ message: string; symbol: string }> {
  return fetchJson(`${API}/api/v1/settings/elements/pending`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, categories }),
  });
}

export async function removePendingElement(symbol: string): Promise<{ message: string }> {
  return fetchJson(`${API}/api/v1/settings/elements/pending/${symbol}`, {
    method: "DELETE",
  });
}

export async function removeSupportedElement(symbol: string): Promise<{ message: string }> {
  return fetchJson(`${API}/api/v1/settings/elements/supported/${symbol}`, {
    method: "DELETE",
  });
}

export async function bootstrapElements(): Promise<{
  bootstrapped: string[];
  failed: string[];
  message: string;
}> {
  return fetchJson(`${API}/api/v1/settings/elements/bootstrap`, {
    method: "POST",
  });
}

// ── Custom Properties ──────────────────────

export async function addCustomProperty(propertyKey: string): Promise<{ message: string; property_key: string }> {
  return fetchJson(`${API}/api/v1/settings/properties`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ property_key: propertyKey }),
  });
}

export async function removeCustomProperty(propertyKey: string): Promise<{ message: string }> {
  return fetchJson(`${API}/api/v1/settings/properties/${propertyKey}`, {
    method: "DELETE",
  });
}

// ── Field Schema Manager ──────────────────────

export interface FieldDefinition {
  name: string;
  data_type: string;
  description: string;
  is_target: boolean;
  is_input: boolean;
  is_required: boolean;
  is_composite_field: boolean;
  category_values: string[];
  aliases: Record<string, string>;
  range_min: number | null;
  range_max: number | null;
  default_value: string | null;
  is_user_added: boolean;
  added_at: string | null;
}

export interface FieldSchemaResponse {
  fields: FieldDefinition[];
}

export async function getFieldSchema(): Promise<FieldSchemaResponse> {
  return fetchJson(`${API}/api/v1/settings/fields`);
}

export async function addUserField(data: {
  name: string;
  data_type: string;
  description?: string;
  is_target?: boolean;
  is_input?: boolean;
  is_composite_field?: boolean;
  category_values?: string[];
  range_min?: number | null;
  range_max?: number | null;
  default_value?: string | null;
}): Promise<{ message: string; field_name: string }> {
  return fetchJson(`${API}/api/v1/settings/fields`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function removeUserField(name: string): Promise<{ message: string }> {
  return fetchJson(`${API}/api/v1/settings/fields/${name}`, {
    method: "DELETE",
  });
}

export async function addFieldCategory(
  fieldName: string, value: string
): Promise<{ message: string; value: string }> {
  return fetchJson(`${API}/api/v1/settings/fields/${fieldName}/categories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
}

export async function removeFieldCategory(
  fieldName: string, value: string
): Promise<{ message: string }> {
  return fetchJson(`${API}/api/v1/settings/fields/${fieldName}/categories/${value}`, {
    method: "DELETE",
  });
}

export async function addFieldAlias(
  fieldName: string, alias: string, canonical: string
): Promise<{ message: string }> {
  return fetchJson(`${API}/api/v1/settings/fields/${fieldName}/aliases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alias, canonical }),
  });
}

export async function exportFieldSchema(): Promise<Record<string, unknown>> {
  return fetchJson(`${API}/api/v1/settings/fields/export`);
}

export async function importFieldSchema(
  schemaData: Record<string, unknown>
): Promise<{ message: string; fields_imported: number; categories_imported: number; aliases_imported: number }> {
  return fetchJson(`${API}/api/v1/settings/fields/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ schema_data: schemaData }),
  });
}

// ── Reset ──────────────────────

export interface ResetResult {
  success: boolean;
  message: string;
  actions_taken: string[];
}

export async function resetElementsAndProperties(): Promise<ResetResult> {
  return fetchJson(`${API}/api/v1/settings/reset/elements`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  });
}

export async function resetAllSettings(): Promise<ResetResult> {
  return fetchJson(`${API}/api/v1/settings/reset/all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  });
}

// ── Models ──────────────────────

export interface SettingsModel {
  id: string;
  display_name: string;
  target: string;
  algorithm: string;
  r2_score: number;
  rmse: number;
  feature_dim: number;
  n_train_samples: number;
  n_test_samples: number;
  is_default: boolean;
  model_file_path: string;
  created_at: string;
}

export async function getSettingsModels(): Promise<SettingsModel[]> {
  return fetchJson(`${API}/api/v1/settings/models`);
}

export async function renameSettingsModel(
  id: string, newName: string
): Promise<{ id: string; display_name: string }> {
  return fetchJson(`${API}/api/v1/settings/models/${id}/rename`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_name: newName }),
  });
}

export async function setDefaultSettingsModel(
  id: string
): Promise<{ id: string; is_default: boolean; target: string }> {
  return fetchJson(`${API}/api/v1/settings/models/${id}/default`, {
    method: "PATCH",
  });
}

export async function deleteSettingsModel(
  id: string
): Promise<{ success: boolean; message: string }> {
  return fetchJson(`${API}/api/v1/settings/models/${id}`, {
    method: "DELETE",
  });
}

export async function batchDeleteModels(
  modelIds: string[]
): Promise<DangerResult> {
  return fetchJson(`${API}/api/v1/settings/models/batch-delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_ids: modelIds }),
  });
}

// ── Danger Zone ──────────────────────

export interface DangerResult {
  success: boolean;
  message: string;
  items_affected: number;
}

export async function purgeAllModels(): Promise<DangerResult> {
  return fetchJson(`${API}/api/v1/settings/danger/purge-models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  });
}

export async function clearPredictionCache(): Promise<DangerResult> {
  return fetchJson(`${API}/api/v1/settings/danger/clear-cache`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  });
}

// ── GNN Status ──────────────────────

export interface GnnStatus {
  enabled: boolean;
  installed: boolean;
  pytorch_version: string;
  chgnet_version: string;
  install_instructions: string;
  message: string;
}

export async function getGnnStatus(): Promise<GnnStatus> {
  return fetchJson(`${API}/api/v1/settings/gnn/status`);
}
