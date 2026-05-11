/**
 * Piezo.AI — Dataset API Client
 * ==============================
 * All HTTP calls for the dataset upload pipeline.
 * Uses fetch() for standard requests, XMLHttpRequest for file upload with progress.
 *
 * Endpoints are defined in apps/api/app/modules/dataset/router.py
 */

import { APP_CONFIG } from "@/lib/constants";

const BASE = `${APP_CONFIG.api.baseUrl}/api/v1/datasets`;

/* ---------- Types ---------- */

export interface DatasetSummary {
  id: string;
  display_name: string;
  original_filename: string;
  status: "pending" | "ready";
  total_rows: number;
  total_columns: number;
  has_composite_fields: boolean;
  uploaded_at: string;
  updated_at: string;
}

export interface DatasetDetail extends DatasetSummary {
  column_mapping: Record<string, string>;
}

export interface MaterialRow {
  id: string;
  uid: number;
  formula: string;
  d33: number | null;
  tc: number | null;
  vickers_hardness: number | null;
  qm: number | null;
  kp: number | null;
  relative_density_pct: number | null;
  sintering_temp_c: number | null;
  sintering_method: string | null;
  ceramic_type: string | null;
  fabrication_method: string | null;
  matrix_type: string;
  filler_wt_pct: number;
  particle_morphology: string;
  particle_size_nm: number | null;
  surface_treatment: string;
  source_doi: string | null;
  source_notes: string | null;
  parse_status: string;
  parse_warnings: string | null;
}

export interface UploadPreview {
  dataset_id: string;
  filename: string;
  columns: string[];
  row_count: number;
  preview_rows: Record<string, unknown>[];
  suggested_mapping: Record<string, string>;
}

export interface DataIssue {
  row_uid: number;
  material_id: string;
  column: string;
  issue_type: string;
  current_value: string | null;
  message: string;
}

export interface ColumnStats {
  column_name: string;
  total_values: number;
  missing_count: number;
  invalid_count: number;
  dtype: string;
}

export interface QualityReport {
  total_rows: number;
  valid_rows: number;
  issue_count: number;
  issues: DataIssue[];
  column_stats: ColumnStats[];
}

export interface PaginatedMaterials {
  items: MaterialRow[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface BackendFieldInfo {
  name: string;
  category: string;
  label: string;
  description: string;
  type: string;
  required: boolean;
  options: string[] | null;
}

export interface BulkUpdateResult {
  updated_count: number;
  deleted_count: number;
  errors: string[];
}

/* ---------- Helper ---------- */

async function apiFetch<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
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
      /* raw text is fine */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

/* ---------- Upload (XHR for progress) ---------- */

export function uploadCSV(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<UploadPreview> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid JSON response from server"));
        }
      } else {
        let detail = xhr.responseText;
        try {
          const json = JSON.parse(xhr.responseText);
          detail = json.detail || detail;
        } catch {
          /* raw text */
        }
        reject(new Error(detail));
      }
    });

    xhr.addEventListener("error", () =>
      reject(new Error("Network error — upload failed")),
    );
    xhr.addEventListener("abort", () =>
      reject(new Error("Upload cancelled")),
    );

    xhr.open("POST", `${BASE}/upload`);
    xhr.send(fd);
  });
}

/* ---------- Column Mapping ---------- */

export function applyMapping(
  datasetId: string,
  mapping: Record<string, string>,
) {
  return apiFetch<DatasetDetail>(`${BASE}/${datasetId}/map`, {
    method: "POST",
    body: JSON.stringify({ mapping }),
  });
}

/* ---------- Quality Report ---------- */

export function getQualityReport(datasetId: string) {
  return apiFetch<QualityReport>(`${BASE}/${datasetId}/quality-report`);
}

/* ---------- Finalize ---------- */

export function finalizeDataset(datasetId: string) {
  return apiFetch<DatasetDetail>(`${BASE}/${datasetId}/finalize`, {
    method: "POST",
  });
}

/* ---------- Dataset CRUD ---------- */

export function listDatasets() {
  return apiFetch<DatasetSummary[]>(`${BASE}/`);
}

export function getDataset(datasetId: string) {
  return apiFetch<DatasetDetail>(`${BASE}/${datasetId}`);
}

export function renameDataset(datasetId: string, displayName: string) {
  return apiFetch<DatasetDetail>(`${BASE}/${datasetId}`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

export function deleteDataset(datasetId: string) {
  return apiFetch<void>(`${BASE}/${datasetId}`, { method: "DELETE" });
}

export function getBackendFields() {
  return apiFetch<BackendFieldInfo[]>(`${BASE}/fields`);
}

/* ---------- Material CRUD ---------- */

export function getMaterials(
  datasetId: string,
  params: {
    search?: string;
    sort_by?: string;
    sort_order?: string;
    page?: number;
    page_size?: number;
  } = {},
) {
  const qs = new URLSearchParams();
  if (params.search) qs.set("search", params.search);
  if (params.sort_by) qs.set("sort_by", params.sort_by);
  if (params.sort_order) qs.set("sort_order", params.sort_order);
  if (params.page) qs.set("page", String(params.page));
  if (params.page_size) qs.set("page_size", String(params.page_size));
  const q = qs.toString();
  return apiFetch<PaginatedMaterials>(
    `${BASE}/${datasetId}/materials${q ? `?${q}` : ""}`,
  );
}

export function addMaterial(
  datasetId: string,
  data: Partial<MaterialRow>,
) {
  return apiFetch<MaterialRow>(`${BASE}/${datasetId}/materials`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateMaterial(
  datasetId: string,
  materialId: string,
  data: Partial<MaterialRow>,
) {
  return apiFetch<MaterialRow>(
    `${BASE}/${datasetId}/materials/${materialId}`,
    { method: "PATCH", body: JSON.stringify(data) },
  );
}

export function bulkUpdateMaterials(
  datasetId: string,
  updates: { id: string; updates: Record<string, unknown> }[],
  deletes: string[],
) {
  return apiFetch<BulkUpdateResult>(`${BASE}/${datasetId}/materials/bulk`, {
    method: "POST",
    body: JSON.stringify({ updates, deletes }),
  });
}

export function clearMaterialColumn(datasetId: string, fieldName: string) {
  return apiFetch<BulkUpdateResult>(`${BASE}/${datasetId}/materials/clear-column/${encodeURIComponent(fieldName)}`, {
    method: "POST",
  });
}

export function deleteMaterial(datasetId: string, materialId: string) {
  return apiFetch<void>(`${BASE}/${datasetId}/materials/${materialId}`, {
    method: "DELETE",
  });
}

/* ---------- Dataset Management (copy, rename, bulk delete) ---------- */

export function copyDataset(datasetId: string, newName?: string) {
  return apiFetch<DatasetDetail>(`${BASE}/${datasetId}/copy`, {
    method: "POST",
    body: JSON.stringify({ new_name: newName }),
  });
}

export function bulkDeleteDatasets(datasetIds: string[]) {
  return apiFetch<{ deleted_count: number; errors: string[] }>(`${BASE}/bulk-delete`, {
    method: "POST",
    body: JSON.stringify({ dataset_ids: datasetIds }),
  });
}
