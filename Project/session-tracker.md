# Piezo.AI v2.1 — Session Progress Tracker

> **Last Updated:** 2026-05-15 | **Total Sessions:** 11 (S0–S9.5)

---

## Session Status

| Session | Feature | Status | Date Started | Date Completed | Notes |
|---------|---------|--------|-------------|----------------|-------|
| **S0** | Cleanup + Scaffold Monorepo + DB Setup | `[x]` Completed | 2026-05-07 | 2026-05-07 | Full scaffold, DB models, dev.sh, version pinning, themes CSS |
| **S1** | Layout Shell (Sidebar, Header, 3 Themes, Routing) | `[x]` Completed | 2026-05-07 | 2026-05-07 | Frontend layout shell, 7 placeholder pages, env-driven branding, responsive 4-tier |
| **S2** | Dataset Upload Pipeline | `[x]` Completed | 2026-05-08 | 2026-05-09 | Backend + Frontend: upload, map, review, explore + edit validation hardening |
| **S3** | ML-Core Foundation | `[x]` Completed | 2026-05-09 | 2026-05-09 | ML-core registry/parsers/features/preprocessing/artifact saver + tests |
| **S4** | Train (Model Studio) | `[x]` Completed | 2026-05-09 | 2026-05-09 | Full ML pipeline + backend + frontend: 8 algos, WebSocket, convergence, terminal, results |
| **S5** | Predict (Unified) + Cross-cutting Fixes | `[x]` Completed | 2026-05-10 | 2026-05-11 | Prediction engine, multi-target batch, formula strict validation, usage prediction engine (11 categories), hardness CI, tabular batch preview, UI/UX fixes |
| **S6** | Dashboard | `[x]` Completed | 2026-05-11 | 2026-05-11 | Stats, quick actions, report generation, model CRUD, dataset management, AI insights, parsed dataset preview, prediction grouping |
| **S7** | Interpretability | `[x]` Completed | 2026-05-12 | 2026-05-14 | SHAP plots, PySR symbolic regression, chart navigation, dev.sh modularization, logging architecture |
| **S8** | Optimization Lab | `[x]` Completed | 2026-05-13 | 2026-05-14 | NSGA-II optimizer, structural analysis, Pareto front, responsive UI, ChartNavigator, dependency management |
| **S9** | Settings + Polish | `[x]` Completed | 2026-05-14 | 2026-05-15 | Settings page, element registry, AI/LLM config, app config, danger zone, GNN status, InfoTooltip, central formula validator, logo upload, undo countdown |
| **S9.5** | Central Field Schema Manager | `[/]` In Progress | 2026-05-15 | — | Central field/schema, multi-cat bug fix, new elements (H,B,N,C,Co,Cr,In,Si,Ni), new categories (X-site, interstitial, network_former), range limit removal, persistence fix, cell edit fix, training deselect sync |

**Legend:** `[ ]` Not Started | `[/]` In Progress | `[x]` Completed | `[!]` Blocked

---

## Session Details

### S0 — Cleanup + Scaffold
- [x] Delete all old code (retain only: .env, .gitignore, resources/, Project/, reference docs)
- [x] Scaffold monorepo: package.json, turbo.json, pnpm-workspace.yaml
- [x] Scaffold apps/api: pyproject.toml, main.py, core/, modules/
- [x] Scaffold apps/web: Next.js 15 init, package.json, app/, components/, lib/
- [x] Scaffold packages/ml-core: pyproject.toml, piezo_ml/ (including registry/ directory)
- [x] Scaffold packages/db: pyproject.toml, models.py (6 tables), alembic init
- [x] Create resources/training-artifacts/ and resources/trained-models/ directories
- [x] Docker compose for PostgreSQL
- [x] dev.sh script (setup/start/stop/clean/db:* with port conflict handling, single Ctrl+C shutdown)
- [x] Update knn_schema_reference.csv with bulk ceramic defaults (already present)
- [x] Verify: `pnpm install`, `pnpm dev` starts without errors
- **Additional S0 work:**
  - [x] `.nvmrc` (Node 20) + `.python-version` (3.13) for version pinning
  - [x] `requirements.txt` — combined Python dependency lock file
  - [x] `README.md` with setup instructions, pyenv/nvm/brew, troubleshooting
  - [x] `globals.css` with all 3 theme color definitions (dark/light/night)
  - [x] Python 3.13 constraint: `requires-python = ">=3.11,<3.14"` in all pyproject.toml (mendeleev requires <3.14)
  - [x] `psycopg2-binary` added for sync Alembic migrations (asyncpg causes PermissionError on macOS Docker)
  - [x] Implementation plan updated: §5.1 Dev Environment + Architecture Rule #9 (Python 3.13 / venv)
  - [x] Auto-generated initial Alembic migration (6 tables matching §5.6 schema)

### S1 — Layout Shell
- [x] Sidebar component with 7 navigation items + collapsible icon-only mode
- [x] Header with system status indicator + theme toggle
- [x] ThemeProvider: dark, light, night (warm amber) themes
- [x] AppShell wrapping all pages
- [x] Route setup for all 7 pages with placeholder content
- [x] globals.css with all 3 theme color definitions
- [x] **Responsive foundation:** 4 breakpoint tiers (XL ≥1440, LG 1080–1439, MD 768–1079, SM <768)
- [x] **Mobile layout:** bottom navigation bar for all 7 sections (hamburger removed per UX decision — bottom nav is more intuitive on mobile)
- [x] Verify: navigate all pages, toggle all 3 themes, check layout at all 4 breakpoints
- **Additional S1 work:**
  - [x] `lib/constants.ts` — central branding config driven by NEXT_PUBLIC_* env variables
  - [x] `.env.example` updated with APP_NAME, APP_LOGO_TEXT, APP_TAGLINE, DEV_NAME/GITHUB/LINKEDIN
  - [x] Sidebar: `position: fixed` + `height: 100vh` (viewport-locked, independent scroll)
  - [x] Collapse animation: icons stay left-aligned, no centering jump (smooth width transition)
  - [x] Mobile header: Piezo.AI logo + dev links + version (replaces hidden sidebar branding)
  - [x] Zustand UI store for sidebar collapsed state
  - [x] SSR-safe `useMediaQuery` hooks (XL/LG/MD/SM breakpoints)
  - [x] Framer Motion animations on sidebar collapse/expand and brand text
  - [x] Created `feature/v2.1.0-s1-layout-shell` branch, hardened `.gitignore`

### S2 — Dataset Upload
- [x] Backend: dataset upload endpoint, DB models (datasets, materials), schema validation
- [x] Backend: **mandatory column mapping** — save dataset in DB with columns renamed to backend field names
- [x] Backend: auto-assign **uid column** (sequential from 1, based on upload order) at DB save time
- [x] Frontend: upload wizard (upload → map → review → explore)
- [x] Frontend: column mapping UI (mandatory, cannot be skipped)
- [x] Frontend: data quality review with issue detection + **unsupported element detection** (formula validation at upload)
- [x] Frontend: **Review Issues CRUD** — inline edit, delete rows/columns, multi-select, shift-select, select-all, Save/Cancel buttons
- [x] Frontend: Dataset explorer table with search, sort, CRUD, overall Save/Cancel, uid column visible
- [x] Frontend: **Parsed Dataset Comparison UI** — side-by-side source vs parsed view, uid-mapped, search, mismatch highlighting
- [x] Multi-dataset support
- [x] Verify: upload piezo_v2.1_test_dataset.csv, map columns, verify uid assigned, verify explorer shows correct data
- **Additional S2 work:**
  - [x] **Safety: validate-on-save + revalidation workflow** — backend validates edits (types + categorical constraints + formula re-validate) before commit; any mutation flips dataset from `ready` → `pending`, UI prompts re-run Review Issues and re-finalize
  - [x] **Premium validation notices (modular):** centralized theme-aligned reusable notice banner surfaces save/validation issues with detailed reasons; invalid edits are reverted to DB state after refresh
  - [x] **Strict bulk vs composite edit validation:** enforce (bulk: matrix=none + filler=0) and (composite: matrix!=none + filler>0 + morphology/size/treatment provided)
  - [x] **Source vs Parsed snapshots for Comparison** — persist `materials.source_row` + `materials.parsed_row` (JSONB) so comparison is true source vs normalized/validated values
  - [x] Review Issues: **Clear/Delete Column** remediation — clear a selected column across all rows (DB-safe “delete column” behavior) and refresh quality report
  - [x] ML-Core: `piezo_ml.registry` (33 elements) + `piezo_ml.validators.formula_validator` (unicode, extraction, validation)
  - [x] Backend: 14 API endpoints, expanded categorical options (3d_print, epoxy, silicone, pvdf_trfe, fluorinated, etc.)
  - [x] Backend: full-text search across ALL material fields (formula, sintering_method, d33, tc, etc.)
  - [x] Frontend: Zustand store with reactive change detection, XHR upload with progress
  - [x] Frontend: DataTable with row edit mode (pencil toggles edit-ready state on all editable cells)
  - [x] Frontend: text wrapping in table cells (break-word, auto-expand row height)
  - [x] Frontend: column borders + left-alignment for all table cells
  - [x] Frontend: ComparisonView rewrite — shows ALL mapped fields in Source/Parsed/Comparison tabs
  - [x] Frontend: column visibility filter + per-field search in ComparisonView
  - [x] Frontend: comparison search now matches only intersection of Search-In + visible columns (prevents hidden-match confusion)
  - [x] Frontend: pagination hardening for external page-size sync and accurate row/page counts
  - [x] Frontend: modular NoticeBanner reused for explorer/review validation feedback and pending workflow notices
  - [x] Frontend: Re-run Review Issues now re-enters wizard review step from pending Explorer datasets
  - [x] Frontend: Review Issues upgraded to full-row table with issue row highlighting + multi-column clear UX (select all/deselect all + guardrails)
  - [x] Review guardrails: non-clearable `uid`/`formula`; block clear if all target metrics (`d33`/`tc`/`vickers_hardness`) would be removed
  - [x] Save/review error details improved: validation errors now surface `uid` with row UUID for clear debugging context
  - [x] Review table issue context restored: row-level issue reasons visible alongside highlighted rows
  - [x] Column clear behavior documented as intentional (`NULL`/sentinel writes), with downstream preprocessing handling required in S4
  - [x] Frontend: deletedIssueIds mapping fix (ReviewIssuesStep delete now works)
  - [x] Test dataset: `piezo_v2.1_test_dataset.csv` — 28 rows covering bulk, composite, edge cases
  - **Note:** Parsed elemental compositions (element fractions, stoichiometry) are S3 scope → ComparisonView shows info banner

### S3 — ML-Core Foundation
- [x] **Central Element Registry** (packages/ml-core/piezo_ml/registry/element_registry.py)
  - 33 supported elements with 27+ pre-computed physics properties each
  - Single source of truth for parser, validator, feature engineer, API
  - **Auto-bootstrap mechanism:** PENDING_ELEMENTS → auto-fetch from mendeleev/pymatgen on startup (in-memory bootstrap for pending symbols) + persisted precomputed registry data source. [Pending-elements UI remains Settings scope in S9.]
- [x] Formula parser (custom multi-phase splitter + chemparse for single phases)
- [x] Formula normalizer (unicode subscripts, garbled dashes, full-width brackets)
- [x] Formula validator (validates against Element Registry, graceful unsupported element handling)
- [x] Feature engineer: elemental mole fractions + weighted physics descriptors from registry
- [x] **Post-parse validation:** verify every field for type mismatches, NaN, out-of-range before training
- [x] Field registry (maps schema fields to ML features)
- [x] Data cleaner (duplicates, types, missing value strategies)
- [x] Data loader and validator
- [x] **Test set policy:** drop rows with missing/invalid values in test set (no imputation), log count
- [x] **Parsed dataset saver** — saves source_with_uid.csv + parsed_compositions.csv with matching uid to resources/training-artifacts/
- [x] Unit tests with pytest
- [x] Verify: parse "K0.5Na0.5NbO3" → correct feature vector
- [x] Verify: parse all formulas in sample_knn_basic.csv → all succeed, parsed_compositions.csv matches expected
- **Additional S3 work:**
  - [x] Added modular parser stack: `formula_normalizer.py` + `formula_parser.py` with top-level multi-phase splitting and leading multipliers
  - [x] Upgraded registry package exports and added property-key API + element lookup helpers
  - [x] Added weighted descriptor stats and structural factors (`tolerance_factor`, `octahedral_factor`) in feature engineering
  - [x] Added preprocessing modules: cleaner, loader, missing-value strategies (knn/mean/median/mode/drop), post-parse validator, orchestration pipeline
  - [x] Enforced split-first preprocessing order for train/test-safe workflow
  - [x] Added robust test-set drop policy log line for missing/invalid values
  - [x] Added parsed dataset artifact writer with stable repo-root path resolution
  - [x] Added pytest coverage for parser, feature vector correctness, and full sample CSV parse/alignment checks
  - [x] Verified `pytest packages/ml-core/tests -q` passes (4 passed)

### S4 — Train
- [x] **S2 sentinel handling in training:** `sentinel_handler.py` distinguishes bulk-ceramic `"none"` (valid) from truly missing data; pre-training validation endpoint returns per-field issues with suggested strategies
- [x] **Only selected fields trigger strategies:** Orchestrator filters missing-value strategies to only apply for user-selected fields/targets — prevents non-selected targets (e.g., vickers_hardness) from dropping all rows via "drop" strategy
- [x] Backend: 9 API endpoints (8 REST + 1 WebSocket) — `GET /algorithms`, `POST /validate`, `POST/GET/DELETE /jobs`, `POST /stop`, `GET /results`, `WS /ws/{id}`
- [x] Backend: **stop endpoint** — propagates `threading.Event` cancel signal through orchestrator → trainer → tuner
- [x] Backend: `TrainingService` with `threading.Thread` background worker, log queue for WebSocket streaming
- [x] ML-Core: `ModelTrainer` with convergence tracking (XGB/LGBM via eval_set, GBR via staged_predict, ANN via loss_curve_)
- [x] ML-Core: `OptunaTuner` — Bayesian hyperparameter optimization with per-algorithm search spaces and cross-validation
- [x] ML-Core: `algorithm_registry.py` — 8 algorithms (XGBoost, RF, SVR, LightGBM, GBR, DT, ANN, Stacking) with full hyperparameter metadata, tooltips, and `build_model()` factory
- [x] ML-Core: **cross-platform multiprocessing** — `platform_utils.py` enforces `n_jobs=1` on macOS, sets `OMP_NUM_THREADS=1`, configures `spawn` start method
- [x] **Model artifact storage** — `.joblib` + metadata JSON to `resources/trained-models/` with dataset UUID traceability
- [x] **Parsed dataset snapshot** — `source_with_uid.csv` + `parsed_compositions.csv` + `feature_vectors.csv` + `preprocessing_log.txt` per training run
- [x] **Graceful parse error handling:** `engineer_dataframe` skips rows with invalid/unsupported formulas, logs warnings with uid + reason, continues training with remaining rows
- [x] **Row alignment after parsing:** Orchestrator aligns train_df/test_df UIDs with feature vectors after skipped rows to prevent dimension mismatch
- [x] Frontend: `PipelineConfigurator` — dataset dropdown, target/field chips, per-field missing value strategy selectors
- [x] Frontend: `AlgorithmSelector` — 8-algorithm card grid with unified/per-target modes and auto-tune toggle
- [x] Frontend: `HyperparameterPanel` — constrained sliders, number inputs, **"i" tooltips** with description + impact + recommended value, reset button
- [x] Frontend: `TrainingTerminal` — dark terminal UI with color-coded logs (info/warning/error/success), auto-scroll, line count
- [x] Frontend: `ProgressBar` — stage-weighted (validate 5% → split 5% → clean 10% → impute 10% → engineer 20% → train 50%), shimmer animation, stage label
- [x] Frontend: `StopButton` — state machine: hidden → play → stop → checkmark, calls backend cancel endpoint
- [x] Frontend: `ConvergenceChart` — Recharts real-time plot for XGBoost/LightGBM/GBR/ANN; info guide explaining good vs bad convergence
- [x] Frontend: `TrainingResults` — R²/RMSE metric cards with color coding (green ≥0.8, yellow 0.5-0.8, red <0.5), train/test counts, duration
- [x] Frontend: `TrainPageContent` — two-panel layout orchestrator (config left + monitor right)
- [x] Frontend: `trainingStore.ts` (Zustand) — pipeline config, job lifecycle, logs, convergence, results
- [x] Frontend: `useTrainingWebSocket` hook — dispatches log/progress/convergence/complete/cancelled/error messages
- [x] Frontend: **WebSocket model normalization** — maps worker format (r2, n_train) to TrainedModelInfo type (r2_score, n_train_samples)
- [x] Frontend: `train.css` — 580 lines of premium CSS with terminal, progress bar, algorithm cards, convergence chart, responsive
- [x] Frontend: defensive null-safe `.toFixed()` calls in TrainingResults to prevent runtime crashes
- [x] Frontend: only auto-selects d33/tc as default targets (not vickers_hardness) to prevent empty-dataset issues
- [x] Verify: train XGBoost/RF on `piezo_v2.1_test_dataset.csv` (28 rows, 3 edge cases skipped) — 2 models produced
- [x] Verify: train RF on `sample_knn_basic.csv` (20 rows) — R²=0.23/0.18 (expected for small dataset)
- [x] Verify: `training-artifacts/` populated with source_with_uid.csv + parsed_compositions.csv + feature_vectors.csv
- [x] Verify: `trained-models/` populated with .joblib + metadata JSON
- [x] Verify: 9 API routes registered, 0 TypeScript errors, 4 S3 pytest tests pass
- [x] **Convergence chart fix:** separate per-target charts with independent Y-axes — prevents scale differences (d33: 0-280, tc: 0-80000) from flattening one target's curve
- [x] **Per-target algorithm selection UX:** sequential highlighting with auto-advance + click any target box to re-highlight and reassign
- [x] **Test datasets:** `piezo_v2.1_test_dataset.csv` (26 rows, 12 edge cases) + `piezo_clean_30.csv` (30 rows, zero issues)
- [x] **Pre-existing S2 TS2352 fixes:** `MaterialRow` → `unknown` → `Record<string, unknown>` cast chain in 3 files
- [x] **Imputation crash fix:** Fallback to `mode` for categorical fields on numeric-only strategies (KNN, Mean) to avoid `boolean value of NA is ambiguous` Pandas crash.
- [x] **Sentinel crash fix:** Safe `int()` casting using `filler_col` summation in `detect_sentinel_issues`.
- [x] **Type-aware missing value UI:** Dynamic dropdowns filtering `allowed_strategies` (e.g., Mode/Drop for Categoricals, KNN/Mean/Drop for Numeric).
- [x] **Regenerated schema-compliant datasets:** Fixed shifted columns for composite rows and added rigorous edge-case test file.
- **Architecture decisions:**
  - Thread-based (not multiprocessing) for safe `threading.Event` cancellation propagation
  - Split-first preprocessing (80/20 before cleaning/imputation) to prevent data leakage
  - Convergence tracking is per-algorithm: XGBoost/LightGBM via eval_set, GBR via staged_predict, ANN via loss_curve_; RF/DT/SVR/Stacking have no natural convergence (by design)
  - Feature columns aligned between train/test splits (reindex with fill_value=0.0)
  - Per-target convergence charts: each target gets its own chart with independent Y-axis domain to prevent scale dominance

### S5 — Predict (Unified)
- [x] ML-Core: `InferenceEngine` — model caching, feature vector construction, prediction with CI estimation via tree ensemble std dev or 10% fallback
- [x] ML-Core: `use_case_mapper.py` — rules-based classification into 6 industrial use-cases (wearables, actuators, transducers, energy harvesting, HT sensors, general)
- [x] ML-Core: Updated `model_saver.py` — now saves `feature_columns` list in metadata JSON for inference reproducibility
- [x] ML-Core: Updated `training_orchestrator.py` — passes `feature_columns` to `save_trained_model`
- [x] ML-Core: Updated `models/__init__.py` — exports `InferenceEngine`, `PredictionResult`, `map_use_case`, `UseCaseResult`, `get_use_case_definitions`
- [x] Backend: `prediction/schemas.py` — Pydantic schemas for single predict, batch predict, formula validation, model management, use-case response
- [x] Backend: `prediction/service.py` — `PredictionService` with single/batch/CSV/dataset prediction, formula validation, model rename/set-default, DB persistence
- [x] Backend: `prediction/router.py` — 9 endpoints: `GET /models`, `PATCH /models/{id}/rename`, `PATCH /models/{id}/default`, `POST /predict`, `POST /predict/batch`, `POST /predict/batch-from-dataset`, `POST /validate-formula`, `GET /supported-elements`, `GET /batch/{id}/download`
- [x] Backend: Registered prediction router at `/api/v1/predictions` in `main.py`
- [x] **Element validation on predict** — unsupported elements → friendly error with supported list
- [x] **Batch prediction** — flags unparseable rows with prediction_status + prediction_notes columns
- [x] **Batch auto-detect:** filler_wt_pct=0 + matrix_type="none" → bulk; else composite
- [x] **Batch from existing dataset** — endpoint to run batch prediction using materials from an existing dataset
- [x] Frontend: `predictions.ts` API client — full typed HTTP client for all prediction endpoints
- [x] Frontend: `predictStore.ts` (Zustand) — state for model selection, formula input/validation, composite params, predictions, comparison list, batch results
- [x] Frontend: `FormulaInput.tsx` — real-time debounced validation with green ✓ / red ✗ / loading spinner, element breakdown, unsupported element warnings
- [x] Frontend: `ModelSelector.tsx` — **per-target model dropdowns** (d33/tc/hardness independently) with glassmorphic menus, auto-select defaults, "Skip" option
- [x] Frontend: `predict/page.tsx` — **multi-model prediction** (runs predictSingle per target, merges results), blocks invalid formulas with reason message, "Added!" comparison confirmation
- [x] Frontend: `predictStore.ts` — updated with `TargetModelSelection`, `comparisonJustAdded` flag
- [x] Frontend: `BatchUpload.tsx` — **added "From Dataset" mode** with dataset picker dropdown alongside CSV upload
- [x] Frontend: `DatasetList.tsx` — **added rename, copy, bulk delete** with select all/deselect all, inline rename, confirmation dialogs
- [x] Frontend: `datasets.ts` API client — added `copyDataset` and `bulkDeleteDatasets` functions
- [x] Frontend: `predict.css` — added per-target dropdown styles, block reason banner, batch mode toggle, comparison "Added!" state (1100+ lines total)
- [x] Frontend: `dataset.css` — added bulk action bar, rename input, checkbox selection styles
- [x] Scripts: `dev.sh` — overhauled: nvm auto-loading, graceful Ctrl+C with explicit frontend/backend/docker stop logging, Docker auto-restart on start, `set -uo pipefail` (no -e), green success banner
- [x] Security: `.gitignore` — added `/resources/prediction-results`
- [x] Plans: `02-cross-cutting-and-build-plan.md` — strengthened no-retry policy + end-of-session reverification rule
- [ ] Verify: predict KNbO3 → d33≈66, tc≈435 is working. Ensure that the trained models are present and choosable in predict section, batch, comparison and normal predictions are working correctly as expected.
- [ ] Verify: predict with unsupported element → shows friendly error, does not crash.

#### S5 Fixes & Enhancements (2026-05-11)
- [x] **Fix: Model DB persistence** — trained models now persist to DB immediately after training via WebSocket loop (not only via GET /jobs/{id}). Predict section model selector now works without manual polling.
- [x] **Fix: Missing value strategies** — context-aware imputation: target fields used as features get full numeric strategies (knn, mean, median, mode, drop); only active training targets are restricted to drop-only. Backend `detect_sentinel_issues` now accepts `targets` parameter. Frontend sends targets in validation request.
- [x] **Fix: Use-case mapper** — replaced flat 0.8/0.98 scoring with smooth Gaussian-based scoring (`exp(-0.5*(x-ideal)²/σ²)`). Added coverage penalty (fewer properties = lower confidence ceiling). All 6 use cases have calibrated ideal centres and sigma spreads.
- [x] **Fix: Single prediction error handling** — removed `break` on first target failure. All targets are now attempted even if one fails. Errors aggregated per-target; partial successes shown.
- [x] **Feature: Strict formula validator** — new `formula_strict.py` module in `validators/`. Enforces: charset restrictions, bracket balance/nesting hierarchy, element token validation (rejects lowercase-only like `k`, `kananb`; multi-lowercase like `Oo`, `Kaaa`; trailing fragments like `KNaNbO3-ooo`). Integrated into `FormulaParser` via `strict_mode` parameter.
- [x] **Feature: Parser mode toggle** — global `strictFormulaMode` in uiStore. Toggle switch on Dataset Management home screen (glassmorphic design). Defaults to strict=ON. `FormulaInput.tsx` reads flag and sends to API.
- [x] **UI: Three-dot kebab menu** — migrated inline Rename/Copy/Delete to a three-dot popover on each dataset card. View button stays visible. Menu has: ✏️ Rename, 📋 Copy (with tick animation), 🗑️ Delete (with confirm). Outside-click dismissal.
- [x] **UI: Manage mode fix** — added `padding-top: 36px` in bulk mode so checkbox doesn't overlap card name. Bulk delete label: "Delete All" (all selected) vs "Delete Selected" (partial).
- [x] **UI: Navigation loader** — `NavigationLoader.tsx` component mounted in AppShell. Shows thin animated gradient bar at viewport top during route transitions. Theme-aware, shimmer animation.
- [x] **Fix: Composite feature training (T1.3)** — Created shared `composite_encoder.py` in `features/`. Training orchestrator now appends 8 composite features (filler_wt_pct, particle_size_nm, 4 categorical encodings, sintering_temp, relative_density) to feature vectors via uid-aligned join. Inference engine refactored to import from shared encoder. **⚠️ Requires model retraining for composite differentiation.**
- [x] **Feature: Usage prediction engine rewrite** — Complete rewrite of `use_case_mapper.py` based on `usage-engineering-logic.md`. 11 research-backed use cases with rule-based scoring (0–100). Confidence tiers: Primary (≥70), Secondary (45–69), Tertiary (30–44). Composite modifiers (+15 wearable, +10 sonar, -20 ultrasonics). Scientific caution notes. Partial-property scaling.
- [x] **UI: UseCaseCard redesign** — tier badges (Highly Recommended/Good Fit/Possible), driving properties as monospace tags, collapsible additional recommendations (top 3–4), scientific caution notes with warning styling.
- [x] **Fix: Duplicate error messages** — `blockReason` no longer repeats FormulaInput's validation errors; shows only actionable hint ("Invalid formula — fix before predicting").
- [x] **Fix: Circular import** — `validators/__init__.py` now uses lazy `__getattr__` for `formula_validator` imports to break parsers↔validators circular dependency.
- [x] Verified: strict mode rejects `kNaNb`, `KNaNbO3-ooo`, `Kaaa` while accepting `K0.5Na0.5NbO3`.
- [x] **Fix: T1.4 — Batch multi-target support** — Rewrote `_run_batch_multi()` to accept `model_ids` dict (`{d33: uuid, tc: uuid, vickers_hardness: uuid}`). Each formula is predicted against ALL selected models. Backend `predict_batch_csv` and `predict_batch_from_dataset` now accept per-target model selection.
- [x] **Fix: T3.4 — Batch tabular results view** — `BatchUpload.tsx` rewritten with inline results table showing per-target columns (d₃₃/Tc/HV) with CI ranges, top use case, score badges, status indicators. Sticky headers, scrollable, error rows highlighted red. Only shows columns for selected targets.
- [x] **Fix: T3.5 — Batch multi-model selection UI** — Frontend sends per-target `model_ids` via FormData JSON. `predictBatchCSV()` and `predictBatchFromDataset()` accept `Record<string, string | null>` model ID dicts.
- [x] **Fix: Hardness CI computation** — `PredictionResult` now includes `hardness_ci_lower/hardness_ci_upper`. `InferenceEngine` computes 95% CI for `vickers_hardness` target (was previously skipped). `PredictionGauges.tsx` displays CI for all three properties.
- [x] **Fix: CSV download extension** — Download filename now always ends with `.csv` (was missing extension for dataset names without `.csv` suffix).
- [x] **Fix: Usage score display** — `UseCaseCard.tsx` uses `getScore()` with fallback: `score → confidence×100`. Tier badges use optional chaining for null safety. `UseCaseInfo` schema extended with `tier/tier_label/driving_properties/score` fields.
- [x] **Schema: Extended response models** — `PredictResponse` now includes `usage_predictions: UsagePredictionsInfo` (recommendations list, caution_notes, property_completeness, properties_used). `BatchPredictSummary` includes `results: BatchResultRow[]` for frontend tabular preview. `BatchPredictRequest` uses `model_ids: dict` instead of single `model_id`.
- [x] **CSV cleanup** — Batch CSV output no longer includes source d33/tc/vickers_hardness columns. Only contains: uid, formula, is_composite, predicted values with CI, top_use_case, use_case_score, status, notes.
- [x] Verify: predict KNbO3 → d33≈66, tc≈435 is working. Ensure that the trained models are present and choosable in predict section, batch, comparison and normal predictions are working correctly as expected.
- [x] Verify: predict with unsupported element → shows friendly error, does not crash.

### S6 — Dashboard
- [x] Backend: `dashboard/schemas.py` — Pydantic schemas for SystemStats, DashboardModel, TargetDistribution, PredictionHistoryItem, ReportGenerateRequest/Response, BulkDeleteModels
- [x] Backend: `dashboard/service.py` — DashboardService with system stats (counts, DB size), model CRUD (rename/delete/set-default/bulk-delete), parsed dataset download, prediction history, report generation orchestration
- [x] Backend: `dashboard/router.py` — 11 endpoints: `GET /stats`, `GET /target-distribution`, `GET /models`, `PATCH /models/{id}/rename`, `PATCH /models/{id}/default`, `DELETE /models/{id}`, `POST /models/bulk-delete`, `GET /models/{id}/parsed-dataset`, `GET /predictions/history`, `POST /reports/generate`, `GET /reports/{id}/download`
- [x] Backend: Registered dashboard router at `/api/v1/dashboard` in `main.py`
- [x] ML-Core: `reporting/chart_generator.py` — Matplotlib chart generator (R²/RMSE comparison, target distribution donut, model performance overview) with Piezo.AI dark theme styling
- [x] ML-Core: `reporting/report_builder.py` — Premium ReportLab PDF builder with branded header/footer, embedded charts, model performance tables, prediction insights section, professional A4 typography
- [x] Frontend: `api/dashboard.ts` — Full typed HTTP client for all 11 dashboard API endpoints
- [x] Frontend: `store/dashboardStore.ts` — Zustand store (fetchAll, fetchStats, fetchModels, renameModel, setDefaultModel, deleteModel, bulkDeleteModels, toggleModelSelection, generateReport)
- [x] Frontend: `StatsCards.tsx` — 4 stat cards (datasets, models, predictions, DB size) with loading skeletons, theme-aware styling
- [x] Frontend: `QuickActions.tsx` — 5 navigation buttons (Train, Predict, Dataset, Optimization Lab, Interpretability) with icons and descriptions
- [x] Frontend: `DatasetList.tsx` — Dataset table with rows/columns count, status badges, View (navigates to Explorer) and Download CSV buttons
- [x] Frontend: `ModelLibrary.tsx` — Model cards grid with **rename** (UUID-stable, inline editing), **delete** (individual with confirm + bulk via Manage mode), **set default** (star icon), **download parsed dataset** button, R²/RMSE/algorithm metrics, target badges, UUID display
- [x] Frontend: `DefaultModelSelector.tsx` — Per-target dropdown to set default model for d33/tc/hardness
- [x] Frontend: `TargetDistributionChart.tsx` — Recharts donut chart with center label showing total model count
- [x] Frontend: `ReportGenerator.tsx` — Checkboxable options (R²/RMSE, performance overview, SHAP [S7], AI insight [requires LLM], material insights), prediction history selector with formula/values/dates
- [x] Frontend: `dashboard/page.tsx` — Dashboard page orchestrator with refresh button, error banner, two-column layout
- [x] Frontend: `dashboard/dashboard.css` — Premium 1050+ line CSS with stats grid, model cards, donut chart, report generator, responsive breakpoints (XL/LG/MD/SM)
- [x] CSS: Imported `dashboard.css` in `globals.css`
- [x] TypeScript: Zero compilation errors verified
- [x] Verify: dashboard shows correct counts after S2-S5 data
- [x] Verify: rename model → name changes everywhere, UUID stays same
- [x] Verify: delete model → removed from list and filesystem
- [x] Verify: set default model → persists across page refresh
- [x] Verify: report PDF generates with charts and branding

#### S6 Fixes & Enhancements (2026-05-11)
- [x] **Fix: Settings in QuickActions** — Added Settings route to QuickActions grid (6 items: Train, Predict, Dataset, Optimization Lab, Interpretability, Settings). Updated grid to 3-column layout.
- [x] **Fix: Dataset View navigation** — DatasetList View button now correctly sets `activeDatasetId` in Zustand datasetStore and navigates to `/dataset`, causing DatasetPage to load the specific dataset in Explorer mode.
- [x] **Fix: Dataset Download CSV** — Replaced broken `/export` endpoint with client-side CSV builder that fetches materials via paginated `/materials` endpoint and generates downloadable CSV blob.
- [x] **Feature: Model download options** — ModelLibrary download button now shows a dropdown with 3 options: (a) Parsed Dataset (.csv), (b) Model Weights (.joblib), (c) Download Both. New backend endpoint `GET /models/{id}/model-file` serves the .joblib file.
- [x] **Rewrite: Premium PDF Report Builder** — Complete rewrite of `report_builder.py` and `chart_generator.py`. Light white theme (was dark), proper column widths using Paragraph-wrapped cells (fixes text overlap), KeepTogether for title-chart co-location, branded indigo header/footer chrome, convergence charts section, usage predictions section, proper R²/RMSE colors, _fit_image() aspect ratio preservation.
- [x] **Fix: Stop button error handling** — StopButton gracefully handles 'already finished' errors by transitioning to checkmark state instead of throwing. Failed states show play button for retry.
- [x] **Feature: Prediction history deletion** — Backend: `DELETE /predictions/{id}` and `POST /predictions/bulk-delete` endpoints. Frontend: individual delete with confirm, bulk delete selected, Zustand store actions. ReportGenerator includes deletion UI.
- [x] **Fix: Dashboard overflow** — Added `overflow-x: hidden` and `max-width: 100%` to dashboard-page container and sections. Model library section allows dropdown overflow.
- [x] **Feature: AI insight notice** — ReportGenerator shows configuration warning when AI insight is enabled, documenting required .env keys (OPENAI_API_KEY / GEMINI_API_KEY).
- [x] **CSS: Download dropdown** — Floating dropdown menu with shadow, border-radius, transition effects. Closes on outside click.
- [x] **CSS: Prediction deletion** — Delete button appears on hover, confirm dialog inline, bulk action bar with count and delete button.
- [x] **Fix: DefaultModelSelector text overlap** — Fixed `.section-description` margin from `-8px` to `2px` top. Added scoped CSS for `.default-selector` with proper title/description spacing and `line-height: 1.4`.
- [x] **Fix: Prediction deletion UX** — Individual delete is now instant (no confirmation). Bulk "Delete Selected" requires Yes/Cancel confirmation dialog. Added Select All / Deselect All toggle button.
- [x] **Fix: AI insight config detection** — Replaced hardcoded "API key not configured" badge with dynamic check via new `GET /dashboard/llm-status` endpoint. Backend reads `LLM_API_KEY` (with `GEMINI_API_KEY` fallback) from Settings. Shows green "✓ google/gemini-3-flash" badge when configured.
- [x] **Fix: d33 rendering in PDF** — Unicode subscript `₃` not in Helvetica font → replaced with HTML `<sub>33</sub>` in ReportLab Paragraph cells. Matplotlib chart labels keep Unicode (rendered as images, no font issue).
- [x] **Fix: PDF chart overflow** — `_fit_image()` now caps max_height to frame height minus 60pt. Performance overview chart uses conditional KeepTogether (≤6 models together, >6 flow naturally). Prevents "too large on page" ReportLab error.
- [x] **Feature: Comparison view parsed data** — DatasetComparisonView now fetches actual `parsed_compositions.csv` from training artifacts via new `GET /dashboard/datasets/{id}/parsed-compositions` endpoint. Source tab shows raw upload, Parsed tab shows elemental decomposition (Na_frac, K_frac, etc.), Comparison tab shows UID-aligned side-by-side. Shows helpful "train first" message when artifacts not found.
- [x] **Backend: GEMINI_API_KEY alias** — Added `GEMINI_API_KEY` field to Settings with `effective_llm_api_key` property that falls back from `LLM_API_KEY` → `GEMINI_API_KEY`.
- [x] **Fix: Parsed dataset enrichment** — Modified `feature_engineer.py` to carry over original properties (d33, tc, hardness, composite params) into `parsed_compositions.csv`. Ensures parsed artifacts act as complete sources of truth, while strictly preserving that training features (`feature_vectors.csv`) exclude labels.
- [x] **Fix: Prediction history UX** — Added `is_composite` badge and composite properties (matrix_type, filler_wt_pct, particle_morphology, particle_size) to prediction history items. Replaced simple date with exact timestamps (HH:MM:SS) in prediction history, Dataset list, and Model library.
- [x] **Feature: PDF Report AI Insights** — Implemented `llm_insights.py` supporting Google, OpenAI, and Ollama. Added `_build_ai_insight_section` to `report_builder.py` that generates AI analysis for overall model performance and specific material applications based on configured LLM credentials. Included rule-based fallback if LLM is unavailable.

### S7 — Interpretability
- [x] ML-Core: `evaluation/shap_analyzer.py` — ShapAnalyzer with TreeExplainer (tree models) + KernelExplainer with safe predict wrapper (Stacking/Voting/SVR). Computes beeswarm (global), waterfall (local), dependence (feature-specific) data. Warning suppression for clean terminal output.
- [x] ML-Core: `evaluation/physics_validator.py` — PhysicsValidator checking SHAP importances vs solid-state physics for d33/Tc/hardness. Threshold-based alignment scoring, confirmed/violation/skipped classification.
- [x] ML-Core: `symbolic_regression/pysr_runner.py` — PySRRunner with graceful Julia/PySR unavailability handling. Equation-to-LaTeX conversion, feature substitution, Pareto front extraction. Optional dep: `pip install 'piezo-ml[symbolic]'`.
- [x] Backend: `interpret/schemas.py` — Pydantic schemas for all 6 endpoints (InterpretModelInfo, ShapBeeswarm/Waterfall/Dependence Request+Response, PhysicsValidation, SymbolicRegression)
- [x] Backend: `interpret/service.py` — InterpretService loading model+data from filesystem, orchestrating SHAP/physics/PySR calls with proper error handling
- [x] Backend: `interpret/router.py` — 6 REST endpoints: `GET /models`, `POST /shap/beeswarm`, `POST /shap/waterfall`, `POST /shap/dependence`, `POST /physics-validation`, `POST /symbolic-regression`
- [x] Backend: Registered interpret router at `/api/v1/interpret` in `main.py`
- [x] Frontend: `lib/api/interpret.ts` — Typed HTTP client for all interpret API endpoints with comprehensive TypeScript interfaces
- [x] Frontend: `lib/store/interpretStore.ts` — Zustand store managing model selection, SHAP beeswarm/waterfall/dependence, physics validation, PySR results with loading/error states
- [x] Frontend: `components/interpret/ModelSelector.tsx` — Model cards with target badges (d₃₃/Tc/Hardness), algorithm, R², sample count, default star
- [x] Frontend: `components/interpret/ShapBeeswarm.tsx` — D3-rendered SVG beeswarm plot (dots color-coded blue→red by feature value), responsive, expand/collapse
- [x] Frontend: `components/interpret/ShapWaterfall.tsx` — Horizontal bar chart (red=positive, blue=negative contributions), sample navigation (prev/next)
- [x] Frontend: `components/interpret/ShapDependence.tsx` — Recharts scatter plot with auto-detected interaction feature coloring, feature dropdown auto-selects most important
- [x] Frontend: `components/interpret/PhysicsValidation.tsx` — Alignment score circle (67%/Strong/Moderate/Weak), confirmed/violation counts, expandable detail rows with physics reasoning
- [x] Frontend: `components/interpret/SymbolicRegression.tsx` — PySR equations with KaTeX rendering, Pareto front chart, equations table, graceful "PySR Not Available" state
- [x] Frontend: `components/interpret/InfoTooltip.tsx` — Reusable click-to-open info popover explaining each chart for non-ML experts
- [x] Frontend: `app/interpret/page.tsx` — Page orchestrator: model selector → beeswarm (full) → waterfall+dependence (half) → physics+PySR (half)
- [x] CSS: `interpret.css` + `interpret-details.css` — Premium styling with model cards, interpret cards, beeswarm SVG, waterfall bars, physics score circle, PySR table, responsive breakpoints (LG/MD/SM)
- [x] CSS: Imported interpret CSS + `katex/dist/katex.min.css` in `globals.css`
- [x] Fix: StackingRegressor SHAP crash — safe predict wrapper with NaN handling, proper KernelExplainer fallback for complex ensemble models
- [x] Fix: Terminal log clutter — suppressed sklearn "fitted without feature names" warnings and numpy RuntimeWarning in SHAP computation
- [x] TypeScript: Zero compilation errors verified
- [x] Verify: run SHAP on trained model → valid beeswarm data, waterfall, dependence, physics validation all working

#### S7 Infrastructure Fixes (2026-05-12)
- [x] **Dev script modularization:** Split monolithic `scripts/dev.sh` (696 lines) into 5 focused library modules under `scripts/lib/`: `_colors.sh` (logging/colors), `_python.sh` (venv/Python), `_node.sh` (Node.js/pnpm), `_database.sh` (DB/migrations), `_network.sh` (connectivity diagnostics). New `dev.sh` is 330 lines. Added `diagnose` command for quick debugging.
- [x] **Network resilience:** `dev.sh` now checks pypi.org reachability before pip install, prompts user before proceeding on failure, and shows network diagnostic info. Pre-installs `setuptools>=75.0` and `wheel` explicitly before package installs to fix build-dependency errors.
- [x] **Missing DB migration added:** `7b2f8e3a1c4d_add_hardness_ci_columns.py` — adds `hardness_ci_lower` and `hardness_ci_upper` to `predictions` table (was specified in S5 but migration was skipped). ORM model `piezo_db/models.py` updated to match.
- [x] **requirements.txt synced:** Added missing `psycopg2-binary` (was in `packages/db/pyproject.toml` but absent from requirements.txt). Added `pysr` under symbolic section. Verified all packages match across `requirements.txt`, `packages/db/pyproject.toml`, `packages/ml-core/pyproject.toml`, and `apps/api/pyproject.toml`.
- [x] **Database schema verified:** 3 migrations (c904b3d initial + 4d1f2f3 material_snapshots + 7b2f8e3 hardness_ci) fully cover the §5.6 schema. ORM models match migration state. All tables, indexes, FKs, and JSONB columns are consistent.

### S8 — Optimization Lab
- [x] Backend: `optimization/schemas.py` — Pydantic schemas for OptimizeRequest, OptimizeResponse, SolutionItem, ConvergencePoint, StructuralRequest/Response
- [x] Backend: `optimization/service.py` — OptimizationService orchestrating NSGA-II optimizer and structural analyzer with trained model loading
- [x] Backend: `optimization/router.py` — 6 REST endpoints: `GET /presets`, `POST /run`, `GET /run/{id}`, `POST /structural/analyze`, `POST /structural/compare`, `GET /supported-elements`
- [x] Backend: Registered optimization router at `/api/v1/optimization` in `main.py`
- [x] ML-Core: `optimization/nsga2_optimizer.py` — NSGA-II multi-objective optimizer using pymoo with trained ML models as surrogate fitness functions. Configurable constraints, generations, population size. Top-level guarded pymoo imports.
- [x] ML-Core: `optimization/pareto_utils.py` — Pareto front extraction, non-dominated sorting, solution ranking, use-case tagging
- [x] ML-Core: `optimization/structural_analyzer.py` — Pymatgen-based structural analysis: tolerance factor, octahedral factor, bond valence, Goldschmidt criteria, perovskite classification, site analysis, physics-based descriptors
- [x] ML-Core: Added `pymoo>=0.6.0` to `pyproject.toml` and `requirements.txt`
- [x] Frontend: `lib/api/optimization.ts` — Full typed HTTP client for all optimization API endpoints
- [x] Frontend: `lib/store/optimizationStore.ts` — Zustand store with FALLBACK_PRESETS, preset selection, run/cancel, solutions, convergence, structural analysis state
- [x] Frontend: `components/optimization/OptimizationConfig.tsx` — Use-case preset selector with configurable target ranges and constraints
- [x] Frontend: `components/optimization/ParetoChart.tsx` — Recharts scatter plot with axis selectors, use-case color coding, ChartNavigator integration
- [x] Frontend: `components/optimization/ConvergenceChart.tsx` — Line chart showing optimization convergence with ChartNavigator controls
- [x] Frontend: `components/optimization/SolutionTable.tsx` — Sortable Pareto-optimal solutions table with rank badges, formula, predicted values, use-case tags, CSV export
- [x] Frontend: `components/optimization/StructuralAnalysis.tsx` — Single/compare mode structural analysis with perovskite classification, site analysis, physics metrics
- [x] Frontend: `components/optimization/ModelSelector.tsx` — Model selection card grid for optimization surrogate models
- [x] Frontend: `app/optimization-lab/page.tsx` — Two-column layout page orchestrator (config left + results right)
- [x] Frontend: `optimization-lab.css` + `optimization-lab-details.css` — Premium CSS with responsive breakpoints, overflow-x handling, min-width constraints
- [x] CSS: Imported optimization lab CSS in `globals.css`
- [x] **Dependency management:** Implemented `pz_check_python_deps()` in `scripts/lib/_python.sh` for automated validation of 18 critical Python dependencies during startup with interactive install prompts
- [x] **Dependency integration:** Integrated dependency check into `scripts/dev.sh` startup sequence

#### S8 Fixes & Enhancements (2026-05-14)
- [x] **Fix: NameError in NSGA-II** — Moved pymoo imports to top-level with try/except guard. `_PiezoOptProblem` class now inherits correctly from aliased `_PymooProblem`.
- [x] **Fix: Preset selector defaulting to custom** — Implemented `FALLBACK_PRESETS` in `optimizationStore.ts` to ensure UI functionality without API-loaded data. Store merges API-fetched presets with hardcoded fallbacks.
- [x] **Fix: S7 badge in ReportGenerator** — Removed legacy `S7` session label from SHAP Analysis Summary checkbox in `ReportGenerator.tsx`.
- [x] **Feature: ChartNavigator component** — Created reusable `components/ui/ChartNavigator.tsx` with GitHub mermaid-style D-pad (4-direction pan + center reset), zoom in/out, expand/collapse fullscreen, copy-as-image/download-PNG/copy-code dropdown, and hide/show controls toggle. Uses native SVG-to-canvas snapshot (no external dependencies). Theme-aware via CSS variables.
- [x] **Feature: ChartNavigator CSS** — `chart-navigator.css` with glassmorphic D-pad buttons, animated copy dropdown, toast feedback, fullscreen expanded mode, responsive breakpoints.
- [x] **Feature: ChartNavigator integration** — Wrapped all 7 chart components across the app with ChartNavigator: ParetoChart, Opt ConvergenceChart, Train ConvergenceChart, TargetDistributionChart, ShapDependence, SymbolicRegression Pareto. Replaced old `ChartNavigation` component usage.
- [x] **Fix: Responsive overflow** — Added `overflow-x: hidden` to `.app-main` and `.page-container`. Added `min-width: 0`, `max-width: 100%`, `overflow-x: auto` to `.opt-card`, `.opt-results-col`, `.opt-layout`. Added `-webkit-overflow-scrolling: touch` to table wrapper. Set `min-width: 600px` on solution table for guaranteed scrollable minimum.
- [x] TypeScript: Zero compilation errors verified
- [x] Verify: All S8 tracker items completed

### S9 — Settings + Polish
- [ ] Frontend: settings page (system env, model library with rename/delete reflecting in DB, API config, danger zone)
- [ ] Frontend: App Environment Configuration UI to manage `APP_VERSION`, names, tags, links, logo, title, favicon, etc. Ensure these are written to `.env` and respect standard priority order.
- [ ] Central new element additional with proper option to add where(if required add multi select option since some element can be added for multiple options). “pending elements UI” belongs to Settings scope so implement it. This help in adding new elements centrally and auto triggering the hardcoding mechanism to fetch and add the element properties into the central property zone for future use which makes startup with minimal dependency. ensure everything gets synced and our codebase knows about these new changes or updates so make everything sync when these changes made, if required add referesh button.
- [ ] Similar central new property/field/parameter addtion and similar other central manager(for frontend, backend, ml core, database, database schema update/migration update due to all these new addition handle by our app with minimal code writing required like in schema changes so a non technical person can also do that without editing the codebase) in setting with nice premium glassy modern ui.  
- [ ] Also include the AI management ui like for local llm support (eg: ollama, llama.cpp, etc and similar other etc), cloud llms like(google, claude, openai, deepseek, coheret, grok etc and similar other etc), other opensource model configuration either cloud or qwen or minimax m series, local or on premisse, other opensource model configuraiton ui, since drop down cannot be enough for these since latest models always comes up and we want all control handed over to user so can add a advanced ui to handle these or like a text box to enter, eg: models, links, api key, provider, model api endpoint, llm engine(eg: ollama, llama.cpp, etc and similar other etc), etc.  
- [ ] Add GNN/CHGNet Transfer Learning	into optimization lab with option for user as optional to install its related dependencies duering setup and start and depedency added to the depedency manager but installed with user permission based on this selectio only in tha tpart where we ask user to accept all or reject all or manually etc during setup, it starts accordingly as we setup and when user goes to ui, it is enabled or disables with proper message as selected by user. Heavy deps (PyTorch). structural analysis. Do logical thinking and reasoning for its and add it to plan before implementing code.
- [ ]  ensure everything gets synced and our codebase knows about these new changes or updates so make everything sync when these changes made, if required add referesh button.
- [ ] Backend: settings endpoints (purge models, clear cache, update config, update `.env` variables)
- [ ] Polish: **responsive design verification** at all 4 breakpoints (XL/LG/MD/SM)
- [ ] Polish: **draggable grid UX** — grip-vertical drag handle, bottom-right corner resize, right-edge card drawer
- [ ] Polish: draggable grids disabled on SM (<768px), static stacked layout fallback
- [ ] Polish: graph expand/zoom across all sections
- [ ] Polish: state persistence (zustand persist)
- [ ] Polish: all 3 themes fully working across all breakpoints
- [ ] Polish: Framer Motion animations, smooth breakpoint transitions (300ms ease)
- [ ] Documentation: README.md, setup guide
- [ ] Verify: full E2E flow works
- [ ] Verify: responsive layout at 1440px, 1080px, 768px, 375px

### S9 — Settings + Polish
- [x] Settings page layout with 6 collapsible sections (System, Elements, AI/LLM, Config, GNN, Danger Zone)
- [x] System Environment panel: stats, Python version, feature flags
- [x] Element Registry: view all 32+ supported elements with category badges (A-site, B-site, dopant, rare_earth, anion)
- [x] Add New Element: input with category pill selector, auto-capitalize, symbol format validation (Xx/Xxx)
- [x] Superheavy element confirmation dialog for 3-character symbols (Uue, Uun etc.)
- [x] Category validation: must select at least one category before adding
- [x] **Element category persistence**: user-selected categories stored in `.settings-customizations.json` and used during classification (fixes "other" bug)
- [x] Bootstrap All: robust error handling per-element, reports successes/failures separately
- [x] Element/property removal with 20-second undo countdown timer (`Undo (Xs)` live countdown)
- [x] Element properties viewer with expandable show more/less, add custom property
- [x] AI/LLM Management: provider selector (Google, OpenAI, Anthropic, DeepSeek, Ollama, Custom), model dropdown
- [x] API key eye-toggle (show/hide) for secure verification
- [x] Temperature/max_tokens sliders with advanced panel
- [x] App Config: unified branding (merged App Branding + Frontend Public into single auto-synced section)
- [x] **Logo upload via API**: backend saves to `/public`, updates `.env` automatically (no manual file placement)
- [x] Premium glassmorphism dropdown toggles for boolean feature flags (ENABLE_GNN, etc.)
- [x] .env import with validation (accepts .env, .env.local, .env.example etc.)
- [x] Danger Zone: clear predictions, clear models, factory reset with confirmation dialogs
- [x] GNN Module status check with installation instructions
- [x] **InfoTooltip**: portal-based, viewport-aware tooltip component with drag handle — replaces all inline tooltips across Settings
- [x] **Central formula validator**: `useFormulaValidation` hook + `FormulaValidationInput` component (reusable across Predict, Optimization, etc.)
- [x] **Crystal Structure Analysis**: integrated FormulaValidationInput with strict mode toggle
- [x] **ChartNavigator restored** on Target Distribution (fixed overflow clipping via percentage-based radii, no external labels)
- [x] Responsive CSS for all settings components (mobile/tablet/desktop)
- [x] Verify: TypeScript compiles clean, Python syntax OK

---

## Blockers & Issues Log

| Date | Session | Issue | Resolution |
|------|---------|-------|------------|
| 2026-05-07 | S0 | Python 3.14 incompatible: `mendeleev` requires `<3.14`, `pymatgen` versions < 2026 require `<3.13` | Fixed: set `requires-python = ">=3.11,<3.14"`, use Python 3.13 via `.python-version` + dev.sh auto-detection |
| 2026-05-07 | S0 | `asyncpg` causes `PermissionError: [Errno 1]` on macOS connecting to Docker PostgreSQL | Fixed: switched Alembic env.py to sync `psycopg2-binary` driver (standard practice, asyncpg only needed at runtime) |
| 2026-05-07 | S0 | `mendeleev>=0.18.0` version doesn't exist (was a typo) | Fixed: corrected to `mendeleev>=0.9.0` (latest is 1.1.0) |
| 2026-05-07 | S1 | `Scatter` icon doesn't exist in lucide-react (500 error on /interpret) | Fixed: replaced with `ScatterChart` which is a valid export |
| 2026-05-07 | S1 | Google Fonts `@import url()` must precede `@import "tailwindcss"` (CSS parse error) | Fixed: moved font imports above tailwind import in globals.css |
| 2026-05-07 | S1 | Sidebar stretches with main content on mid-screens (footer goes to bottom of scroll) | Fixed: `position: fixed` + `height: 100vh`, margin-left on header/main via `data-sidebar-collapsed` attribute |
| 2026-05-07 | S1 | Collapse animation: icons jump to center during transition (bad UX) | Fixed: removed `justify-content: center` in collapsed state, icons stay left-aligned |
| 2026-05-07 | S1 | Bottom nav doesn't fill full width on mobile (gap on right side) | Fixed: `flex: 1 1 0` + `justify-content: space-evenly` for even distribution |
| 2026-05-08 | S2 | Alembic migration failed from `packages/db` with `Path doesn't exist: packages/db/alembic` | Fixed: set `script_location = %(here)s/alembic` in `packages/db/alembic.ini`; documented working invocation patterns in README |
| 2026-05-09 | S4 | `uid` column missing when DataFrame passed directly to orchestrator (not via CSV loader) | Fixed: auto-insert `uid` column (1..N) if absent from input DataFrame |
| 2026-05-09 | S4 | Train/test feature columns misaligned — different frac_* columns from different formula sets | Fixed: reindex test features to match train columns with `fill_value=0.0` |
| 2026-05-09 | S4 | `vickers_hardness` auto-selected as field → "drop" strategy removes all rows (field is 100% missing) | Fixed: only auto-select d33/tc as targets; filter strategies to only apply for selected fields |
| 2026-05-09 | S4 | `r2_score.toFixed()` crash — WebSocket sends `r2` but component expects `r2_score` | Fixed: normalize WebSocket model data in hook + add defensive null checks with `?? 0` |
| 2026-05-09 | S4 | `engineer_dataframe` crashes on invalid/unsupported formulas in dataset | Fixed: try/except in `engineer_dataframe`, skip rows with parse errors, log warnings with uid + reason |
| 2026-05-09 | S4 | Dimension mismatch after skipping unparseable rows (feature_vectors ≠ train_df length) | Fixed: align train_df/test_df to surviving UIDs after feature engineering |
| 2026-05-09 | S4 | Convergence chart: shared Y-axis — d33 (0-280) flattened when tc (0-80000) dominates scale | Fixed: separate per-target charts with independent Y-axis domains |
| 2026-05-09 | S4 | Per-target algo selection: all clicks assign to first target (d33) | Fixed: sequential highlighting with auto-advance + clickable target boxes for re-assignment |
| 2026-05-09 | S4 | TS2352: `MaterialRow` to `Record<string, unknown>` cast error in 3 files | Fixed: cast through `unknown` first (`m as unknown as Record<...>`) |
| 2026-05-09 | S4 | Sentinel Handler `TypeError: int()` crash | Fixed: Switched from trying to cast boolean mask directly to `int()` to summing it correctly |
| 2026-05-09 | S4 | `boolean value of NA is ambiguous` during imputation | Fixed: Categorical `pd.NA` converted to `None`, auto-fallback to Pandas mode imputation if KNN/Mean/Median selected for non-numeric fields |
| 2026-05-09 | S4 | Strategy dropdown confusing for categorical fields | Fixed: Backend `FieldIssue` returns `allowed_strategies`, UI dynamically filters `<select>` options |
| 2026-05-09 | S4 | Composite datasets had shifted columns | Fixed: Regenerated `piezo_clean_30.csv` and `piezo_v2.1_test_dataset.csv` with precise `pd.DataFrame` explicit mapping |
| 2026-05-11 | S5 | Trained models not visible in Predict section — `TrainedModel` DB rows only inserted via `GET /jobs/{id}`, but frontend never calls it after WebSocket completion | Fixed: Persist completed job directly from WebSocket loop in `training/router.py` after streaming ends. `GET /jobs/{id}` kept as safety net |
| 2026-05-11 | S5 | d33/tc/hardness only show "drop" for missing values — `_get_allowed_strategies()` always checked `TARGET_FIELDS` first, ignoring context | Fixed: Made strategies context-aware: pass `targets` param to `detect_sentinel_issues`. Fields in `TARGET_FIELDS` but not in active targets get full numeric strategies (knn/mean/median/mode/drop) |
| 2026-05-11 | S5 | Use-case confidence plateau at 0.80/0.98 — `_score_use_case` returns flat 0.8 for unbounded ranges (`max: 9999`) | Fixed: Rewrote with smooth Gaussian scoring + coverage penalty for partial property predictions |
| 2026-05-11 | S5 | First target failure stops all predictions — `break` in predict page loop | Fixed: Removed `break`, added per-target try/catch, aggregated errors for partial success display |
| 2026-05-11 | S5 | Invalid formulas like `kNaNb`, `KNaNbO3-ooo`, `Kaaa` accepted by parser | Fixed: New `formula_strict.py` validator + integrated into `FormulaParser` with `strict_mode` toggle |
| 2026-05-11 | S5 | Dataset card actions cluttered — Rename/Copy/Delete inline overflow | Fixed: Migrated to three-dot kebab popover menu with outside-click dismissal |
| 2026-05-11 | S5 | Manage mode checkbox overlaps dataset name | Fixed: Added `padding-top: 36px` for `.dataset-card.bulk-mode` |
| 2026-05-11 | S5 | No visual feedback during page navigation | Fixed: `NavigationLoader.tsx` with animated gradient bar, mounted in AppShell |
| 2026-05-11 | S5 | Batch prediction only predicts d33 (single model_id) | Fixed: Rewrote batch pipeline to accept per-target `model_ids` dict, predicts ALL selected targets per formula |
| 2026-05-11 | S5 | Batch CSV includes irrelevant source columns (d33, tc, vickers_hardness) | Fixed: Clean CSV output with only uid, formula, predicted values + CI, use case, status |
| 2026-05-11 | S5 | No tabular preview of batch results — only summary + download | Fixed: `BatchUpload.tsx` shows inline results table with per-target columns, CI ranges, use case + score |
| 2026-05-11 | S5 | Hardness prediction has no confidence interval (CI null) | Fixed: Added `hardness_ci_lower/upper` to `PredictionResult`, CI computation for `vickers_hardness` target in inference engine |
| 2026-05-11 | S5 | Usage score shows `undefined%` beside use case name | Fixed: `getScore()` fallback in UseCaseCard, `score` field added to `UseCaseInfo` schema and router |
| 2026-05-11 | S5 | Download filename missing `.csv` extension | Fixed: Router appends `.csv` if not present in download filename |
| 2026-05-11 | S6 | Parsed Preview not showing parsed data (ComparisonView expected training artifacts) | Fixed: Rewrote `DatasetComparisonView.tsx` to call on-demand `POST /dashboard/datasets/{id}/parse` directly. No training required. |
| 2026-05-11 | S6 | Dashboard Kebab Menu Overflow | Fixed: Removed `overflow: hidden` from `.dashboard-page` and `.dashboard-section` to allow dropdowns to escape bounds. |
| 2026-05-11 | S6 | No quick way to select/deselect all targets/features in Training Config | Fixed: Added glassmorphism Select All / Deselect All buttons in `PipelineConfigurator.tsx` with proper target labels (d₃₃, Tc, Hardness). |
| 2026-05-11 | S6 | Reports saw d33, tc, hardness predictions for same formula as 3 separate materials | Fixed: Implemented 10-second time window grouping by formula in `service.py`. Unified item with `member_ids` allows correct bulk deletion and merged report rendering. |
| 2026-05-11 | S6 | PDF Reports lacked context and had generic metadata | Fixed: Updated `report_builder.py` with Piezo.AI metadata, descriptive section paragraphs, and formula-merged prediction insight table. |
| 2026-05-12 | S7 | `pip install` failing: "Could not find a version for setuptools>=75.0" on `packages/db` | Root cause: DNS failure resolving pypi.org. Fixed: dev.sh pre-installs build deps, checks network before install, shows diagnostics. Also split dev.sh into 5 modular lib files for maintainability. |
| 2026-05-14 | S7 | `pnpm` v10 from Homebrew crashing on Node 20 | Fixed: Updated `_node.sh` to explicitly test `pnpm --version`. If it crashes due to Node mismatch, auto-reinstalls `pnpm@9.15.4` locally. |
| 2026-05-14 | S7 | Backend logs cluttered with noisy queries/HTTP requests or hidden entirely | Fixed: Revamped `logging_config.py`. Console is clean (defaults to `INFO`), but heavy loggers (SQL/HTTP/ML) are strictly routed to a new `backend_detailed_{timestamp}.log`. Added `PZ_LOG_LEVEL` environment switch to override. |
| 2026-05-14 | S7 | Terminal logs lost after shutdown | Fixed: Updated `dev.sh start` to use process substitution (`tee`) to capture both frontend and backend stdout/stderr into a persistent `session_{timestamp}.log` file. |
| 2026-05-14 | S7 | Dependency installation globally pollutes laptop | Fixed: Updated `dev.sh` setup to prompt for "Accept/Reject All" global dependencies. Selecting "Reject All" now forcefully isolates Python by cloning `pyenv` directly into `$ROOT_DIR/.pyenv`. |
| 2026-05-13 | S8 | `NameError: name 'Problem' is not defined` in NSGA-II optimizer | Fixed: Moved pymoo imports to top-level with try/except guard. `_PiezoOptProblem` now inherits from `_PymooProblem` alias. |
| 2026-05-13 | S8 | Optimization preset selector always defaults to "custom" | Fixed: Added `FALLBACK_PRESETS` in `optimizationStore.ts` with hardcoded presets. Store merges API-fetched with fallbacks. |
| 2026-05-13 | S8 | Charts/tables cropped on medium/small screens — no scroll, data truncated | Fixed: Added `overflow-x: auto/hidden` to `.app-main`, `.page-container`, `.opt-card`, `.opt-results-col`. Added `min-width: 600px` on table, `-webkit-overflow-scrolling: touch` on table wrapper. |
| 2026-05-14 | S8 | Legacy S7 badge visible in ReportGenerator | Fixed: Removed `<span className="report-badge-future">S7</span>` from `ReportGenerator.tsx`. |
| 2026-05-14 | S8 | No standardized chart controls (pan/zoom/expand/copy) across sections | Fixed: Created reusable `ChartNavigator.tsx` with D-pad, zoom, fullscreen, copy-as-image (native SVG-to-canvas), download PNG, hide/show. Integrated across all 7 chart components. |
| 2026-05-15 | S9.5 | Element Registry multi-category bug: adding C as B-site+dopant only shows first category | Fixed: Backend returns `categories: list[str]` instead of single `category: str`. Frontend shows primary badge + `+N` tooltip for additional categories. |
| 2026-05-15 | S9.5 | Element additions lost on server restart — `SUPPORTED_ELEMENTS` frozenset doesn't include user-added | Fixed: `_load_or_bootstrap_registry()` now reads `.settings-customizations.json` at startup to include user-added elements. |
| 2026-05-15 | S9.5 | d33, tc, hardness, sintering_temp_c had artificial max limits (3000, 1500, 2000, 2000) | Fixed: Set `range_max=None` in field_schema_manager for all 4 fields. No upper bound now. |
| 2026-05-15 | S9.5 | Cell editing in DataTable silently reverted due to HTML5 `type="number"` validation | Fixed: Switched to `type="text"` with `inputMode="decimal"`. No browser-imposed validation on blur. |
| 2026-05-15 | S9.5 | Deselecting a target in PipelineConfigurator didn't clean up missingStrategies/validationIssues | Fixed: `toggleTarget` and `toggleField` now explicitly clear both `missingStrategies` and `validationIssues` for deselected fields. |
| 2026-05-15 | S9.5 | Factory reset didn't clear `element_categories` from customizations | Fixed: `reset_elements_and_properties()` now saves empty `element_categories: {}` too. |
