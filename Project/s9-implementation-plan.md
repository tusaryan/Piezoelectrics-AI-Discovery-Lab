# S9 — Settings + Polish Implementation Plan

> **Session:** S9 | **Scope:** Settings Page, Polish, Extras | **Date:** 2026-05-14

---

## Overview

S9 is the final implementation session covering: Settings page (full-stack), global polish (responsive, draggable grids, animations, state persistence), and extras (AI/LLM management, GNN dependency scaffolding, element/property management).

## Implementation Phases

### Phase 1: Backend Settings Module

### Phase 2: Frontend Settings API + Store

### Phase 3: Frontend Settings UI Components

### Phase 4: Polish (responsive, animations, persistence)

### Phase 5: Documentation & Verification

See detailed plans in the codebase files themselves.

# S9.5 — Completion Plan (Remaining Tasks)

> **Date:** 2026-05-15 | **Status:** In Progress

---

## Completed Items ✅

1. Field Schema Manager in ML-Core (field_schema_manager.py)
2. New element categories (X_SITE, INTERSTITIAL, NETWORK_FORMER)
3. New elements added: H, B, N, C, Co, Cr, In, Si, Ni
4. Backend API for element registry returns `categories` list
5. Frontend PendingElements with +N badge for multi-category display
6. TARGET_STRATEGIES expanded in sentinel_handler.py (knn, mean, median, mode, drop)
7. Frontend category colors for all categories

---

## Remaining Tasks (T1-T14)

### T1: Training Store — Remove Missing Strategy/Validation Issue Actions

- **File:** `apps/web/lib/store/trainingStore.ts`
- Add `removeMissingStrategy(field)` action
- Add `removeValidationIssue(field)` action
- Used for dynamic deselect sync when user deselects a field

### T2: Dataset Cell Editing — Remove Validation on Cell Edit

- **File:** `apps/web/components/dataset/ReviewIssuesStep.tsx` + DataTable
- Currently: edits get validated and silently revert if invalid
- Fix: Make cells simple text edit, remove per-cell validation
- Keep re-validation feature at save/row level (not cell level)
- Changes should persist and show as edited (highlighted)
- Row-level validation still applies

### T3: Dataset Re-validation After Edit

- **File:** `apps/web/lib/store/datasetStore.ts`
- When existing saved dataset is edited, show "Dataset needs re-validation" prompt
- "Re-run Review Issues" button to re-validate entire dataset
- Mark as "needs re-validation" after any edit to existing data

### T4: Remove Max Limits for Numerical Fields

- **File:** `packages/ml-core/piezo_ml/registry/field_schema_manager.py`
- Change d33, tc, vickers_hardness, sintering_temp_c range_max from values to `None` (unlimited)
- Allow +infinity for predictions and dataset values

### T5: Schema CSV Update

- **File:** `resources/sample-and-test-dataset/material_schema_reference.csv`
- Add new categories from WS4:
  - fabrication_method: screen_printing, bridgman, flux_growth, hydrothermal, sputtering, czochralski, solid_state, sol_gel, solution_cast, spin_coating, 3d_printing
  - sintering_method: spark_plasma, hot_pressing, cold_sintering, liquid_phase, none
  - matrix_type: polyurea, plla, pla, polyurethane
  - surface_treatment: oleic_acid, hydrogen_peroxide
  - particle_morphology: nanowire, nanosheet, tube, unknown

### T6: Alias Editing UI for Fields

- **File:** `apps/web/components/settings/FieldSchemaManager.tsx`
- Add UI to edit/add/remove aliases for any categorical field
- Show current aliases, add new alias mapping, remove aliases

### T7: Field Editing UI (Range, Type, Categories)

- **File:** `apps/web/components/settings/FieldSchemaManager.tsx`
- Allow editing existing field properties:
  - Range min/max for numeric fields
  - Description
  - Category values (add/remove for categorical fields)
  - Default values

### T8: Save to Main Codebase Option

- **Files:** `apps/api/app/modules/settings/service.py` + Frontend
- Add "Save to Codebase" button in settings UI
- When user makes changes (new elements, new fields, new categories, aliases), show option to save to main codebase
- Save to: element_registry_data.json, .field-customizations.json (committed to repo)
- Currently only saves to user-specific files (.settings-customizations.json)
- Need: Add export path to codebase resources folder

### T9: Properties Per Element UI Improvements

- **File:** `apps/web/components/settings/PendingElements.tsx`
- Improve UI for "properties per element" section
- Make it more intuitive and user-friendly
- Better layout, clearer labels, easier to add/remove properties

### T10: Verify All Features Working

- Run through all previous S9.5 features
- Verify nothing is broken
- Verify all features from original plan

---

## Implementation Order

1. T4 (Numerical limits) - Quick fix
2. T1 (Training store actions)
3. T5 (Schema CSV)
4. T2 + T3 (Dataset cell editing fixes)
5. T6 + T7 (Field editing UI)
6. T8 (Save to codebase)
7. T9 (Properties UI improvement)
8. T10 (Verification)

# S9.5 — Central Field/Schema Manager + Training Fixes

> **App:** Piezo.AI | **Version:** 2.1.1 | **Date:** 2026-05-15

---

## Overview

This session implements a **Central Field/Schema Manager** — a single source of truth for all material fields, their data types, allowed categories, ranges, and aliases. Similar to the Central Element Registry, this manager centralizes field definitions so adding a new field or category propagates automatically across the entire stack (Frontend, Backend, ML-Core, DB).

---

## Work Streams

| #   | Work Stream                         | Scope                                                      |
| --- | ----------------------------------- | ---------------------------------------------------------- |
| WS1 | Central Field Manager — ML-Core     | New `field_schema_manager.py` in `registry/`               |
| WS2 | Central Field Manager — Backend API | New endpoints + service functions                          |
| WS3 | Central Field Manager — Frontend UI | New Settings section with premium UI                       |
| WS4 | Default Field/Category Additions    | Add new categories to existing fields + aliases            |
| WS5 | Element Registry Bug Fix            | Multi-category selection showing only first                |
| WS6 | New Element Additions               | C, N, H with properties                                    |
| WS7 | Schema CSV Update                   | Update `material_schema_reference.csv`                     |
| WS8 | Factory Reset Enhancement           | Reset user-added fields too                                |
| WS9 | Training Missing Value Fix          | All strategies for d33/tc/hardness + dynamic deselect sync |

---

## WS1: Central Field Manager — ML-Core

**File:** `packages/ml-core/piezo_ml/registry/field_schema_manager.py`

### Data Model

```python
@dataclass
class FieldDefinition:
    name: str                    # e.g., "fabrication_method"
    data_type: str               # "float"|"int"|"string"|"category"
    description: str
    is_target: bool
    is_required: bool
    is_composite_field: bool
    category_values: list[str]   # For category type only
    aliases: dict[str, str]      # {"pvdf-trfe": "p_vdf_trfe"}
    range_min: float | None
    range_max: float | None
    default_value: str | None
    is_user_added: bool
    added_at: str | None
```

### Key Design

1. **Default fields** hardcoded in `DEFAULT_FIELD_SCHEMA` dict
2. **User additions** persisted in `resources/.field-customizations.json`
3. Runtime: defaults + user additions merged into `FIELD_SCHEMA`
4. All modules import from here: `field_options_registry.py`, `field_registry.py`, `composite_encoder.py`, `sentinel_handler.py`

### API Functions

- `get_field_schema()`, `add_user_field()`, `remove_user_field()`
- `add_category_value()`, `remove_category_value()`, `add_alias()`
- `export_field_schema()`, `import_field_schema()`, `reset_field_schema()`
- `resolve_alias()` — Apply alias mapping during data ingestion

---

## WS2: Central Field Manager — Backend API

### New Endpoints (extend settings router)

| Method   | Path                                         | Description               |
| -------- | -------------------------------------------- | ------------------------- |
| `GET`    | `/settings/fields`                           | Get complete field schema |
| `POST`   | `/settings/fields`                           | Add new user field        |
| `DELETE` | `/settings/fields/{name}`                    | Remove user-added field   |
| `POST`   | `/settings/fields/{name}/categories`         | Add category value        |
| `DELETE` | `/settings/fields/{name}/categories/{value}` | Remove user category      |
| `POST`   | `/settings/fields/{name}/aliases`            | Add alias mapping         |
| `POST`   | `/settings/fields/export`                    | Export schema as JSON     |
| `POST`   | `/settings/fields/import`                    | Import schema from JSON   |

---

## WS3: Central Field Manager — Frontend UI

**File:** `apps/web/components/settings/FieldSchemaManager.tsx`

Premium glassmorphic UI with:

1. **Current Fields Table** — field name, type badge, category/range info, user-added indicator, remove button for user fields
2. **Add New Field Form** — name validation (no spaces, lowercase, `_` separated), type selector, conditional UI (category: chip input; numeric: range min/max)
3. **Add Category to Existing Field** — Quick-add for existing category fields
4. **Import/Export** — Export/import field schema JSON for portability

---

## WS4: Default Field/Category Additions

### New Categories

| Field                 | New Values                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fabrication_method`  | `screen_printing`, `bridgman`, `flux_growth`, `hydrothermal`, `sputtering`, `czochralski`, `solid_state`, `sol_gel`, `solution_cast`, `spin_coating`, `3d_printing` |
| `sintering_method`    | `spark_plasma`, `hot_pressing`, `cold_sintering`, `liquid_phase`, `none`                                                                                            |
| `matrix_type`         | `polyurea`, `plla`, `pla`, `polyurethane`                                                                                                                           |
| `surface_treatment`   | `oleic_acid`, `hydrogen_peroxide`                                                                                                                                   |
| `particle_morphology` | `nanowire`, `nanosheet`, `tube`, `unknown`                                                                                                                          |

### New Aliases

| Field               | Alias         | Maps To      |
| ------------------- | ------------- | ------------ |
| `matrix_type`       | `pvdf-trfe`   | `p_vdf_trfe` |
| `matrix_type`       | `p(vdf-trfe)` | `p_vdf_trfe` |
| `surface_treatment` | `untreated`   | `none`       |

---

## WS5: Element Registry Multi-Category Bug Fix

**Problem:** Adding "C" with B-site + dopant only shows first category.

**Root Cause:** `service.py:get_element_registry()` line 409: `cat = stored_cats[0]` only takes first.

**Fix:**

1. Backend: Return `categories: list[str]` in element info
2. Schema: Add `categories` field to `ElementInfo`
3. Frontend: Show first category + `+N` hover tooltip for additional categories
4. Sync element to ALL selected classification sets

---

## WS6: New Element Additions (C, N, H)

1. Add to `SUPPORTED_ELEMENTS` frozenset in `element_registry.py`
2. Add to classification sets in `element_classification.py` (C→DOPANT, N→DOPANT, H→DOPANT)
3. Bootstrap to populate `element_registry_data.json`

---

## WS7: Schema CSV Update

Update `resources/sample-and-test-dataset/material_schema_reference.csv` with all new categories/values from WS4.

---

## WS8: Factory Reset Enhancement

Add `reset_field_schema()` to `reset_all_settings()` — deletes `.field-customizations.json` and reloads defaults.

---

## WS9: Training Missing Value + Deselect Sync Fixes

### Fix 1: Allow all strategies for d33/tc/hardness

**Change:** `sentinel_handler.py` — Change `TARGET_STRATEGIES = ("drop",)` to `TARGET_STRATEGIES = ("knn", "mean", "median", "mode", "drop")`

### Fix 2: Dynamic deselect sync

**Changes to `PipelineConfigurator.tsx`:**

- Filter displayed issues: `validationIssues.filter(i => selectedFields.includes(i.field))`
- Add X button to each strategy row for manual removal
- On field deselect: remove from `missingStrategies` and `validationIssues`

**Changes to `trainingStore.ts`:**

- Add `removeMissingStrategy(field)` and `removeValidationIssue(field)` actions

---

## Sync Matrix

| Layer          | Syncs Via                                         |
| -------------- | ------------------------------------------------- |
| ML-Core        | `field_schema_manager.py` imported by all modules |
| Backend        | Service calls `field_schema_manager` directly     |
| Frontend       | Fetches `/settings/fields` API                    |
| Dataset Upload | Reads schema at validation time                   |
| Training       | Dynamic field selection from schema               |
| Prediction     | Composite dropdowns from schema                   |
| DB             | No migration needed — fields are JSONB/VARCHAR    |

---

## Implementation Order

1. WS1 → WS4 → WS6 → WS5 → WS2 → WS3 → WS9 → WS7 → WS8

# S9.5 Implementation Plan — Part 1: Core Fixes & Enhancements

> **App:** Piezo.AI | **Version:** 2.1.1 | **Date:** 2026-05-15

---

## Task Tracker

| #   | Task                                                          | Status | Priority |
| --- | ------------------------------------------------------------- | ------ | -------- |
| T1  | Element Registry multi-category bug fix                       | `[x]`  | P0       |
| T2  | New element additions (H,B,N,C,Co,Cr,In,Si,Ni) + categories   | `[x]`  | P0       |
| T3  | Element category persistence across server restart            | `[x]`  | P0       |
| T4  | Remove max limits on d33, tc, hardness, sintering_temp_c      | `[x]`  | P0       |
| T5  | Cell edit persistence fix                                     | `[x]`  | P0       |
| T6  | Re-validation workflow verification                           | `[x]`  | P1       |
| T7  | Training deselect sync + X button                             | `[x]`  | P0       |
| T8  | New element categories (X-site, interstitial, network_former) | `[x]`  | P1       |
| T9  | Alias management UI in settings                               | `[ ]`  | P1       |
| T10 | Field/category editing (edit existing defaults)               | `[ ]`  | P1       |
| T11 | Save to main codebase feature                                 | `[ ]`  | P1       |
| T12 | Properties per element UI improvement                         | `[ ]`  | P1       |
| T13 | Factory reset includes field schema                           | `[x]`  | P1       |
| T14 | Schema CSV update                                             | `[ ]`  | P2       |
| T15 | Session tracker update                                        | `[x]`  | P2       |

## Implementation Order: T4 -> T1 -> T2+T8 -> T3 -> T5 -> T7 -> T9 -> T11 -> T12 -> T13 -> T14 -> T15

# S9.5 Implementation Summary — Changes Made

> **Date:** 2026-05-15 | **Session:** S9.5 Central Field/Schema Manager

---

## Completed Tasks (10/15)

### T4: Remove Max Range Limits ✅

**File:** `packages/ml-core/piezo_ml/registry/field_schema_manager.py`

- d33: `range_max=3000` → `range_max=None`
- tc: `range_max=1500` → `range_max=None`
- vickers_hardness: `range_max=2000` → `range_max=None`
- sintering_temp_c: `range_max=2000` → `range_max=None`

### T1: Element Registry Multi-Category Bug Fix ✅

**Files changed:**

- `apps/api/app/modules/settings/service.py` — `get_element_registry()` now returns `categories: list[str]` alongside `category: str`
- `apps/api/app/modules/settings/schemas.py` — Added `categories: list[str]` to `ElementInfo`, `available_categories` to `ElementRegistryResponse`
- `apps/web/lib/api/settings.ts` — Updated TypeScript interfaces
- `apps/web/components/settings/PendingElements.tsx` — Shows primary badge + `+N` tooltip for multi-category elements

### T2+T8: New Elements + Categories ✅

**Files changed:**

- `packages/ml-core/piezo_ml/registry/element_registry.py` — Added H, B, N, C, Co, Cr, In, Si, Ni to `SUPPORTED_ELEMENTS`, updated `B_SITE` and `DOPANTS` sets
- `packages/ml-core/piezo_ml/registry/element_classification.py` — Added X_SITE, INTERSTITIAL, NETWORK_FORMER categories; added new elements to DOPANT and B_SITE categories; added convenience frozensets and `get_element_categories()` function; updated `COORDINATION_NUMBERS`
- `apps/api/app/modules/settings/service.py` — Updated `DEFAULT_SUPPORTED_ELEMENTS`

### T3: Element Persistence Across Restart ✅

**File:** `packages/ml-core/piezo_ml/registry/element_registry.py`

- `_load_or_bootstrap_registry()` now reads `.settings-customizations.json` at startup to include user-added elements in addition to the default `SUPPORTED_ELEMENTS` frozenset

### T5: Cell Edit Persistence Fix ✅

**File:** `apps/web/components/dataset/DataTable.tsx`

- Changed cell edit inputs from `type="number"` to `type="text"` with `inputMode="decimal"` to prevent HTML5 validation from silently reverting edits on blur

### T7: Training Deselect Sync ✅

**File:** `apps/web/components/train/PipelineConfigurator.tsx`

- `toggleField()` now clears both `missingStrategies` and `validationIssues` for deselected fields
- `toggleTarget()` now properly removes deselected targets from `selectedFields` and cleans up strategies/issues

### T13: Factory Reset Fix ✅

**File:** `apps/api/app/modules/settings/service.py`

- `reset_elements_and_properties()` now includes `element_categories: {}` in the reset payload

### T15: Session Tracker Update ✅

**File:** `Project/session-tracker.md`

- Added 6 bug fix entries for S9.5
- Updated S9.5 status summary

---

## Remaining Tasks (5/15)

| Task | Description                                              | Priority |
| ---- | -------------------------------------------------------- | -------- |
| T9   | Alias management UI in settings                          | P1       |
| T10  | Field/category editing (edit existing defaults)          | P1       |
| T11  | Save to main codebase feature                            | P1       |
| T12  | Properties per element UI improvement                    | P1       |
| T14  | Schema CSV update (create material_schema_reference.csv) | P2       |

---

## Recommended Git Commit Message

```
feat(S9.5): element registry multi-category, new elements & categories, range/edit/sync fixes

ML-Core:
- Add 9 new elements (H, B, N, C, Co, Cr, In, Si, Ni) to SUPPORTED_ELEMENTS
- Add 3 new lattice categories: X-site/anion, interstitial, network_former
- Add get_element_categories() for multi-category lookup
- Remove artificial range_max on d33/tc/hardness/sintering_temp_c
- Fix server restart persistence: load user-added elements from customizations file

Backend:
- Fix multi-category bug: return categories[] instead of single category string
- Return available_categories from element classification for dynamic UI
- Fix factory reset to clear element_categories from customizations

Frontend:
- Show multi-category badges with +N hover tooltip in Element Registry
- Fix cell edit persistence: switch from type=number to type=text
- Fix training deselect sync: clean up missingStrategies and validationIssues
- Dynamic category pills from backend available_categories
- Add new category colors for X-site, interstitial, network_former
```
