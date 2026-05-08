/**
 * Piezo.AI — Dataset Store (Zustand)
 * ====================================
 * State management for the dataset upload wizard, material CRUD,
 * selection, and editing flows.
 */

import { create } from "zustand";
import type {
  DatasetSummary,
  DatasetDetail,
  MaterialRow,
  DataIssue,
  QualityReport,
  UploadPreview,
  BackendFieldInfo,
} from "@/lib/api/datasets";

/* ---------- Types ---------- */

export type WizardStep = "upload" | "map" | "review" | "explore";

interface DatasetState {
  /* Wizard */
  wizardStep: WizardStep;
  wizardActive: boolean;

  /* Active dataset */
  activeDatasetId: string | null;
  activeDataset: DatasetDetail | null;

  /* Upload */
  uploadedFile: File | null;
  uploadProgress: number;
  isUploading: boolean;
  uploadError: string | null;

  /* Preview (from upload response) */
  csvColumns: string[];
  rowCount: number;
  previewRows: Record<string, unknown>[];
  suggestedMapping: Record<string, string>;

  /* Column mapping */
  columnMapping: Record<string, string>;
  isMappingSaving: boolean;
  mappingError: string | null;

  /* Backend field metadata (from /fields endpoint) */
  backendFields: BackendFieldInfo[];

  /* Quality report */
  qualityReport: QualityReport | null;
  isLoadingReport: boolean;

  /* Materials (explorer) */
  materials: MaterialRow[];
  totalMaterials: number;
  currentPage: number;
  pageSize: number;
  totalPages: number;
  isLoadingMaterials: boolean;
  searchQuery: string;
  sortBy: string;
  sortOrder: "asc" | "desc";

  /* Selection */
  selectedIds: Set<string>;
  lastSelectedId: string | null;

  /* Editing */
  editedCells: Map<string, Record<string, unknown>>;
  pendingDeletes: Set<string>;
  newRows: MaterialRow[];

  /* Dataset list */
  datasets: DatasetSummary[];
  isLoadingDatasets: boolean;

  /* Comparison tab */
  comparisonTab: "source" | "parsed" | "comparison";
}

interface DatasetActions {
  /* Wizard */
  setWizardStep: (step: WizardStep) => void;
  startWizard: () => void;
  enterWizardAtStep: (step: WizardStep) => void;

  /* Active dataset */
  setActiveDatasetId: (id: string | null) => void;
  setActiveDataset: (ds: DatasetDetail | null) => void;

  /* Upload */
  setUploadedFile: (file: File | null) => void;
  setUploadProgress: (pct: number) => void;
  setIsUploading: (v: boolean) => void;
  setUploadError: (err: string | null) => void;

  /* Preview */
  setUploadPreview: (preview: UploadPreview) => void;

  /* Column mapping */
  setColumnMapping: (mapping: Record<string, string>) => void;
  updateMapping: (csvCol: string, backendField: string) => void;
  removeMapping: (csvCol: string) => void;
  setIsMappingSaving: (v: boolean) => void;
  setMappingError: (err: string | null) => void;

  /* Backend fields */
  setBackendFields: (fields: BackendFieldInfo[]) => void;

  /* Quality report */
  setQualityReport: (report: QualityReport | null) => void;
  setIsLoadingReport: (v: boolean) => void;

  /* Materials */
  setMaterials: (items: MaterialRow[], total: number, page: number, totalPages: number) => void;
  setIsLoadingMaterials: (v: boolean) => void;
  setSearchQuery: (q: string) => void;
  setSorting: (col: string, order: "asc" | "desc") => void;
  setCurrentPage: (page: number) => void;
  setPageSize: (size: number) => void;

  /* Selection */
  toggleRowSelection: (id: string) => void;
  selectRange: (ids: string[]) => void;
  selectAll: () => void;
  deselectAll: () => void;
  setLastSelectedId: (id: string | null) => void;

  /* Editing */
  editCell: (materialId: string, field: string, value: unknown) => void;
  markForDeletion: (ids: string[]) => void;
  unmarkForDeletion: (ids: string[]) => void;
  addNewRow: (row: MaterialRow) => void;
  removeNewRow: (tempId: string) => void;
  discardChanges: () => void;

  /* Dataset list */
  setDatasets: (datasets: DatasetSummary[]) => void;
  setIsLoadingDatasets: (v: boolean) => void;
  removeDatasetFromList: (id: string) => void;

  /* Comparison */
  setComparisonTab: (tab: "source" | "parsed" | "comparison") => void;

  /* Full reset */
  resetWizard: () => void;
  resetStore: () => void;
}

/* ---------- Derived helpers ---------- */

export function getHasUnsavedChanges(state: DatasetState): boolean {
  return (
    state.editedCells.size > 0 ||
    state.pendingDeletes.size > 0 ||
    state.newRows.length > 0
  );
}

export function getChangesSummary(state: DatasetState): string {
  const parts: string[] = [];
  if (state.editedCells.size > 0)
    parts.push(`${state.editedCells.size} edit${state.editedCells.size > 1 ? "s" : ""}`);
  if (state.newRows.length > 0)
    parts.push(`${state.newRows.length} new row${state.newRows.length > 1 ? "s" : ""}`);
  if (state.pendingDeletes.size > 0)
    parts.push(`${state.pendingDeletes.size} deletion${state.pendingDeletes.size > 1 ? "s" : ""}`);
  return parts.join(", ");
}

/* ---------- Initial state ---------- */

const initialState: DatasetState = {
  wizardStep: "upload",
  wizardActive: false,
  activeDatasetId: null,
  activeDataset: null,
  uploadedFile: null,
  uploadProgress: 0,
  isUploading: false,
  uploadError: null,
  csvColumns: [],
  rowCount: 0,
  previewRows: [],
  suggestedMapping: {},
  columnMapping: {},
  isMappingSaving: false,
  mappingError: null,
  backendFields: [],
  qualityReport: null,
  isLoadingReport: false,
  materials: [],
  totalMaterials: 0,
  currentPage: 1,
  pageSize: 50,
  totalPages: 1,
  isLoadingMaterials: false,
  searchQuery: "",
  sortBy: "uid",
  sortOrder: "asc",
  selectedIds: new Set(),
  lastSelectedId: null,
  editedCells: new Map(),
  pendingDeletes: new Set(),
  newRows: [],
  datasets: [],
  isLoadingDatasets: false,
  comparisonTab: "source",
};

/* ---------- Store ---------- */

export const useDatasetStore = create<DatasetState & DatasetActions>(
  (set, get) => ({
    ...initialState,

    /* --- Wizard --- */
    setWizardStep: (step) => set({ wizardStep: step }),
    startWizard: () =>
      set({
        wizardActive: true,
        wizardStep: "upload",
        activeDatasetId: null,
        activeDataset: null,
        uploadedFile: null,
        uploadProgress: 0,
        uploadError: null,
        csvColumns: [],
        previewRows: [],
        suggestedMapping: {},
        columnMapping: {},
      }),
    enterWizardAtStep: (step) =>
      set({
        wizardActive: true,
        wizardStep: step,
      }),

    /* --- Active dataset --- */
    setActiveDatasetId: (id) => set({ activeDatasetId: id }),
    setActiveDataset: (ds) => set({ activeDataset: ds }),

    /* --- Upload --- */
    setUploadedFile: (file) => set({ uploadedFile: file, uploadError: null }),
    setUploadProgress: (pct) => set({ uploadProgress: pct }),
    setIsUploading: (v) => set({ isUploading: v }),
    setUploadError: (err) => set({ uploadError: err }),

    /* --- Preview --- */
    setUploadPreview: (p) =>
      set({
        activeDatasetId: p.dataset_id,
        csvColumns: p.columns,
        rowCount: p.row_count,
        previewRows: p.preview_rows,
        suggestedMapping: p.suggested_mapping,
        columnMapping: p.suggested_mapping, // pre-fill with suggestions
      }),

    /* --- Column mapping --- */
    setColumnMapping: (mapping) => set({ columnMapping: mapping }),
    updateMapping: (csvCol, backendField) =>
      set((s) => ({
        columnMapping: { ...s.columnMapping, [csvCol]: backendField },
      })),
    removeMapping: (csvCol) =>
      set((s) => {
        const m = { ...s.columnMapping };
        delete m[csvCol];
        return { columnMapping: m };
      }),
    setIsMappingSaving: (v) => set({ isMappingSaving: v }),
    setMappingError: (err) => set({ mappingError: err }),

    /* --- Backend fields --- */
    setBackendFields: (fields) => set({ backendFields: fields }),

    /* --- Quality report --- */
    setQualityReport: (report) => set({ qualityReport: report }),
    setIsLoadingReport: (v) => set({ isLoadingReport: v }),

    /* --- Materials --- */
    setMaterials: (items, total, page, totalPages) =>
      set({
        materials: items,
        totalMaterials: total,
        currentPage: page,
        totalPages,
      }),
    setIsLoadingMaterials: (v) => set({ isLoadingMaterials: v }),
    setSearchQuery: (q) => set({ searchQuery: q, currentPage: 1 }),
    setSorting: (col, order) =>
      set({ sortBy: col, sortOrder: order, currentPage: 1 }),
    setCurrentPage: (page) => set({ currentPage: page }),
    setPageSize: (size) => set({ pageSize: size, currentPage: 1 }),

    /* --- Selection --- */
    toggleRowSelection: (id) =>
      set((s) => {
        const next = new Set(s.selectedIds);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return { selectedIds: next, lastSelectedId: id };
      }),
    selectRange: (ids) =>
      set((s) => {
        const next = new Set(s.selectedIds);
        ids.forEach((id) => next.add(id));
        return { selectedIds: next };
      }),
    selectAll: () =>
      set((s) => ({
        selectedIds: new Set(s.materials.map((m) => m.id)),
      })),
    deselectAll: () => set({ selectedIds: new Set(), lastSelectedId: null }),
    setLastSelectedId: (id) => set({ lastSelectedId: id }),

    /* --- Editing --- */
    editCell: (materialId, field, value) =>
      set((s) => {
        const tempRowIdx = s.newRows.findIndex((r) => r.id === materialId);
        if (tempRowIdx !== -1) {
          const nextNewRows = [...s.newRows];
          nextNewRows[tempRowIdx] = {
            ...nextNewRows[tempRowIdx],
            [field]: value as never,
          };
          return { newRows: nextNewRows };
        }

        const originalRow = s.materials.find((m) => m.id === materialId);
        if (!originalRow) {
          return {};
        }

        const normalize = (v: unknown) => {
          if (v === "") return null;
          return v ?? null;
        };

        const originalValue = normalize((originalRow as Record<string, unknown>)[field]);
        const nextValue = normalize(value);

        const nextEdited = new Map(s.editedCells);
        const current = { ...(nextEdited.get(materialId) || {}) };

        if (Object.is(originalValue, nextValue)) {
          delete current[field];
        } else {
          current[field] = value;
        }

        if (Object.keys(current).length === 0) {
          nextEdited.delete(materialId);
        } else {
          nextEdited.set(materialId, current);
        }

        return { editedCells: nextEdited };
      }),
    markForDeletion: (ids) =>
      set((s) => {
        const pending = new Set(s.pendingDeletes);
        const edited = new Map(s.editedCells);
        let newRows = [...s.newRows];

        ids.forEach((id) => {
          if (id.startsWith("__new_")) {
            newRows = newRows.filter((r) => r.id !== id);
          } else {
            pending.add(id);
            edited.delete(id);
          }
        });

        return { pendingDeletes: pending, editedCells: edited, newRows };
      }),
    unmarkForDeletion: (ids) =>
      set((s) => {
        const next = new Set(s.pendingDeletes);
        ids.forEach((id) => next.delete(id));
        return { pendingDeletes: next };
      }),
    addNewRow: (row) =>
      set((s) => ({ newRows: [...s.newRows, row] })),
    removeNewRow: (tempId) =>
      set((s) => ({
        newRows: s.newRows.filter((r) => r.id !== tempId),
      })),
    discardChanges: () =>
      set({
        editedCells: new Map(),
        pendingDeletes: new Set(),
        newRows: [],
        selectedIds: new Set(),
      }),

    /* --- Dataset list --- */
    setDatasets: (datasets) => set({ datasets }),
    setIsLoadingDatasets: (v) => set({ isLoadingDatasets: v }),
    removeDatasetFromList: (id) =>
      set((s) => ({
        datasets: s.datasets.filter((d) => d.id !== id),
        activeDatasetId: s.activeDatasetId === id ? null : s.activeDatasetId,
      })),

    /* --- Comparison --- */
    setComparisonTab: (tab) => set({ comparisonTab: tab }),

    /* --- Resets --- */
    resetWizard: () =>
      set({
        wizardStep: "upload",
        wizardActive: false,
        uploadedFile: null,
        uploadProgress: 0,
        uploadError: null,
        csvColumns: [],
        rowCount: 0,
        previewRows: [],
        suggestedMapping: {},
        columnMapping: {},
        mappingError: null,
        qualityReport: null,
        editedCells: new Map(),
        pendingDeletes: new Set(),
        newRows: [],
        selectedIds: new Set(),
      }),
    resetStore: () => set(initialState),
  }),
);
