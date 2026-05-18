# Piezo.AI v2.1 — Implementation Plan (Part 1: Architecture & Section Features)

> **App:** Piezo.AI | **Version:** 2.1.0 | **Updated:** 2026-05-11

---
01-architecture-and-sections.md, 02-cross-cutting-and-build-plan.md, session-tracker.md, s9-implementation-plan.md
## 1. Monorepo Structure

```
Piezoelectrics-AI-Discovery-Lab/
├── apps/
│   ├── api/                    # FastAPI — DUMB PIPE ONLY (no ML logic)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── core/           # config, database, errors, logging
│   │   │   └── modules/        # dataset/, training/, prediction/, optimization/, interpret/, settings/
│   │   ├── pyproject.toml
│   │   └── alembic/
│   └── web/                    # Next.js 16 + React 19
│       ├── app/                # Pages: dashboard, dataset, train, predict, optimization-lab, interpret, settings
│       ├── components/         # Organized by section + shared ui/ and layout/
│       ├── lib/                # api/, store/, hooks/, utils
│       └── package.json
├── packages/
│   ├── ml-core/                # ALL ML logic lives here
│   │   └── piezo_ml/
│   │       ├── registry/       # CENTRAL element registry (single source of truth)
│   │       ├── parsers/        # formula parsing, validation, normalization
│   │       ├── features/       # feature engineering (reads from registry)
│   │       ├── pipeline/       # data loading, cleaning, validation, training
│   │       ├── models/         # inference engine, model registry, composite predictor, use-case mapper
│   │       ├── evaluation/     # SHAP analyzer, metrics
│   │       ├── optimization/   # NSGA-II Pareto front, crystal structure analysis
│   │       ├── symbolic_regression/  # PySR integration
│   │       └── reporting/      # PDF report generation
│   └── db/                     # SQLAlchemy models + Alembic migrations
├── scripts/                    # dev.sh, setup.sh, migrate.sh
├── resources/                  # datasets, schema reference, UI previews (RETAINED)
│   ├── training-artifacts/     # Parsed datasets + preprocessing logs per training run
│   └── trained-models/         # Model .joblib files + metadata JSON
├── Project/                    # plans, session tracker (RETAINED)
├── docker/
├── .env / .env.example / .gitignore
├── package.json / turbo.json / pnpm-workspace.yaml
└── README.md
```

> [!IMPORTANT]
> **Architectural Rule:** FastAPI is a DUMB PIPE. Zero ML logic in `apps/api/`. All ML computations, model loading (joblib), formula parsing, feature engineering, and training live exclusively in `packages/ml-core/`.

---

## 2. Sections & Navigation (7 Sections)

| #   | Section          | Route               | Sidebar Icon | Description                                                        |
| --- | ---------------- | ------------------- | ------------ | ------------------------------------------------------------------ |
| 1   | Dashboard        | `/dashboard`        | BarChart3    | System overview, quick actions, report generation, model mgmt      |
| 2   | Dataset          | `/dataset`          | Database     | Upload, map, clean, review, explore CSV datasets                   |
| 3   | Train            | `/train`            | BrainCog     | Configure ML pipelines, train models, convergence, results         |
| 4   | Predict          | `/predict`          | Zap          | **Unified** prediction: bulk ceramics + composites + hardness      |
| 5   | Optimization Lab | `/optimization-lab` | FlaskConical | Crystal structure analysis + multi-objective property optimization |
| 6   | Interpretability | `/interpret`        | Eye          | SHAP (beeswarm, waterfall, dependence), Symbolic Regression (PySR) |
| 7   | Settings         | `/settings`         | Settings     | Models library, system env, API config, danger zone                |

---

## 3. Feature Specifications Per Section

### 3.1 Dashboard

- **Stats cards:** datasets uploaded (count, rows, columns, status: ready/pending), trained models count, predictions made, training jobs
- **Dataset list:** view all datasets with total rows & columns per dataset, delete individually or all, status badges (ready/pending), a **View** button to open the dataset in a tabular view in Dataset Explorer with full CRUD (cell/row level) with an overall save/cancel button and reflect those changes in DB for consistency, sorting, searching, and option to download as CSV
- **Quick actions:** buttons navigating to Train, Predict, Optimization Lab, Interpretability
- **Default model selector:** choose/set which trained model is used for prediction
- **Model target distribution:** donut chart (d33, tc, hardness percentages)
- **Trained models list:** all models with model_id (internal UUID — never changes even if renamed), algorithm, **rename option** (updates display name in DB, UUID stays constant), targets (d33, tc, hardness), R², RMSE, date trained, training duration. Delete option (individual, multi-select, or all). **Download parsed dataset** button per model — downloads the final parsed dataset used for training (same row order as source, includes uid column mapping to source rows) for manual verification
- **Report generation:** checkboxable options:
  - R²/RMSE curves
  - Predicted vs actual graphs (d33, tc, hardness — if applicable based on trained targets)
  - AI insight about model performance (optional — only if AI/LLM server is configured; if not configured, show alert: "AI insights unavailable — configure LLM in Settings")
  - Specific material prediction insight — applications, use-cases, and relevant properties. Displayed in tabular format. **Predictions for the same formula are grouped and merged** into a single row to provide a unified material insight.
  - SHAP analysis summary
  - **Report quality:** PDF must look modern, sleek, premium — proper headings, subheadings, spacing, **contextual descriptions for every section**, embedded charts/graphs, **Piezo.AI branded metadata and footer**, organized layout
  - Downloadable PDF
- **Refresh button:** fetches latest DB info (models, datasets, stats)
- **System online/offline status** indicator in header

### 3.2 Dataset Upload & Management

- **Upload:** drag-and-drop CSV only, file size indicator, progress bar
- **Column mapping (mandatory — cannot be skipped):** map each CSV column to backend schema fields. Once mapped, save the dataset in DB with column names **renamed to the backend field names** so no extra mapping overhead exists in other sections:
  - formula, d33, tc, vickers_hardness, qm, kp, relative_density_pct, sintering_temp_c, sintering_method, ceramic_type, fabrication_method
  - Composite fields: matrix_type, filler_wt_pct, particle_morphology, particle_size_nm, surface_treatment
  - Traceability: source_doi, source_notes
- **Bulk ceramic defaults:** For bulk ceramic rows (filler_wt_pct=0), composite-specific fields must default to standard sentinel values:
  - matrix_type → "none"
  - particle_morphology → "none"
  - surface_treatment → "none"
  - particle_size_nm → null/NA
  - This must also be reflected in `knn_schema_reference.csv` in resources
- **Bulk vs composite validation (S2 hardening):**
  - Bulk row must satisfy: `matrix_type='none'` AND `filler_wt_pct=0`
  - Composite row must satisfy: `matrix_type!='none'` AND `filler_wt_pct>0`
  - Composite-compulsory descriptor fields during user edits: `particle_morphology`, `particle_size_nm`, `surface_treatment` (use `untreated` if no treatment is applied)
  - Invalid edits are rejected with reason and row is reverted to DB value
- **Parsed Dataset Comparison UI (S6 hardening)** — side-by-side source vs parsed view, uid-mapped, search, mismatch highlighting. **Parsing is performed on-demand in real-time** from database materials, allowing users to verify stoichiometry and features immediately after upload, without waiting for a training job.
- **Review Issues table UX (S2 hardening):** full dataset table is visible during review. Rows with issues are highlighted in red, edited cells keep orange highlight, and users can clear one or multiple columns at once with select-all/deselect-all helpers. `uid` and `formula` are non-clearable and at least one target metric among `d33` / `tc` / `vickers_hardness` must remain. **Save/Cancel buttons** to commit or reject all changes. All changes persist to DB for consistency.
- [missed in S2 need to include in S7]**Multi-select** with shift-select for range selection, select-all with deselect-specific. 
- **Column clear semantics (S2 hardening):** clear action writes storage-safe missing markers (numeric/text nullable fields → `NULL`; sentinel categorical composite fields → `none`; `filler_wt_pct` → `0` for bulk-safe fallback). UI renders these as `—` or sentinel text. This is intentional and must be treated as missing/sentinel during S4 preprocessing.
- **Edit safety + re-validation (S2 hardening):** user edits in Review/Explorer are **validated server-side** (numeric type checks, categorical option constraints, formula re-validation). Any successful mutation on a dataset that was previously `ready` flips it back to `pending` so the user must re-run Review Issues and re-finalize. This prevents silent divergence from preprocessing expectations.
- **Central modular alerting (S2 hardening):** save/validation/review notices use a reusable themed banner component (info/success/warning/error) shared across Explorer and Review screens for consistent premium UX.
- **Future extension note (post-S2):** if later sessions need ceramic powder/ceramic surface processing metadata independent of polymer composites, introduce separate fields (do not overload `surface_treatment` semantics used for composite filler treatment in S2).
- **Data quality report:** breakdown of issues per column, count of valid/invalid rows
- **Dataset Explorer:** virtualized tabular view, sort by any column, search formulas, CRUD on rows/cells. **Overall Save/Cancel buttons** (not cell-level auto-save). Dataset shown in source upload order. **uid column** (auto-assigned sequential integer starting from 1, assigned at DB save time based on original upload order — NOT the same as DB row index). uid is used ONLY for reference between source and parsed datasets, not for training. If a row is dropped during preprocessing, its uid is preserved in the parsed dataset so users can trace back to the original source row
- **Explorer hardening (S2):** explicit top-level validation alert after save attempts, and pending datasets expose a working **Re-run Review Issues** action that re-enters wizard review mode directly.
- **Pagination hardening (S2):** page-size dropdown (25/50/100) is externally controlled from store state so list size and page navigation remain consistent after edits/search/sort.
- **Multi-dataset support:** upload multiple datasets, view/select any from dashboard
- **Status tracking:** ready (wizard completed), pending (wizard not completed)
- **Navigation:** Upload Another, View Dashboard, Train Models buttons
- **"Start Over"** to reset the upload wizard

### 3.3 Train (Model Studio)

- **Dataset selector:** dropdown to switch between uploaded datasets
- **Field selector:** choose which fields to use for training based on what's available in the selected dataset. formula + d33 + tc are compulsory. Others (vickers_hardness, qm, composite fields) are optional and only selectable if present in dataset
- **Bulk vs composite auto-detection:** filler_wt_pct=0 + matrix_type="none" → bulk ceramic; else composite. This determines which features are engineered
- **Data preprocessing pipeline (all in ml-core):**
  - Train/test split FIRST (80/20) before any transformation. **Test set policy:** do NOT impute missing/invalid values in the test set — instead drop those rows and log the count in terminal (e.g., "Test set: dropped 3/8 rows with missing values")
  - Clean dataset (remove exact duplicates, fix data types)
  - Per-field missing/invalid data handling: user chooses per field from dropdown:
    - KNN imputer (default for numerical)
    - Mean / Median / Mode
    - Drop row
    - Smart defaults auto-set based on field type (numerical → KNN/median, categorical → mode)
  - Inputs coming from S2 clear/remediation (e.g., "—", `NULL`, `none`, bulk sentinels) are explicitly recognized before training so user-selected strategies can be applied consistently.
  - Feature engineering: parse chemical formula → elemental fractions + physics descriptors (atomic mass, radius, electronegativity, valence electrons, tolerance factor). **Post-parse validation:** verify every field in every row after parsing to catch type mismatches, NaN values, or out-of-range data before training begins
  - Normalization/scaling
  - Since we allow download of parsed dataset so we should also add feature to directly upload the parsed dataset and add a training pipelines for this. Since this dataset is already clean so it will not have any issue like missing/invalid value and we can skip parsing and other relevant step for training with this special case.
- **ML algorithm selection:** XGBoost, Random Forest, SVM/SVR, LightGBM, Gradient Boosting, Decision Tree, ANN, Stacking
- **Per-field model assignment:** option to use one algorithm for all targets OR select specific algorithm per target (e.g., XGBoost for d33, RF for tc, GBR for hardness)
- **Fine-tuning parameters:** dynamically displayed based on selected algorithm. Each parameter has:
  - Slider/number input with current value, **constrained to the valid range** for that parameter (e.g., max_depth: 1–50, n_estimators: 10–5000)
  - Recommended default value shown
  - **"i" tooltip** with: brief description of what the parameter does, how increasing/decreasing it affects model performance (e.g., "More trees → more robust but slower, risk of overfitting if max_depth is also high"), guidance for when to alter it, **recommended standard starting value** shown
  - **Auto-tune toggle** (Optuna) to automatically find optimal hyperparameters
- **Execution queue:** add multiple training jobs for sequential execution
- **Convergence chart:** real-time plotting of actual model convergence metric (not fake 1/x curve). Has an **"i" info button** on hover that shows: what a good convergence graph looks like vs bad, with visual examples and guidance on when to stop training
- **Terminal UI:** real-time backend log streaming showing:
  - Preprocessing steps being executed
  - Train/test split info (rows, columns)
  - Rows before and after cleaning, dropped rows count
  - Feature engineering progress
  - Training progress per algorithm
  - Errors (red), Warnings (yellow), Success (green) with color coding
  - Same logs appear in actual backend Python terminal — all ML-Core process logs are mirrored
- **Progress bar:** real-time training completion percentage (NOT mock/random). Advances step-by-step after each ML stage completes (e.g., split → clean → engineer → train). Each stage's weight is proportional to its expected duration, with status of current work text updated above the progress bar so that user knows what is currently being processed.
- **Stop button:** abort training mid-process — signal is **propagated to the actual backend ML process** (not just frontend). Icon state machine: hidden (no pipeline) → play (pipeline configured) → stop (training in progress) → checkmark (success). Resets to play when a new pipeline is queued
- **Results on success:**
  - Predicted vs Actual scatter plots for d33, tc, hardness (where applicable)
  - R²/RMSE metric graphs
  - Cross-model comparison: compare metrics across multiple trained models with bar charts
- **Report inclusion:** option to include training graphs and multi-model comparisons in the PDF report
- **n_jobs=1** enforced for tree-based models on macOS to prevent segfaults. Handle all multiprocessing-related issues (OpenMP, fork vs spawn). **Cross-platform:** compatible with macOS (Sequoia 15.7.5+), Windows 11, and Linux
- **Mode selector:** Manual (user sets params) vs Auto (Optuna-tuned) training modes

### 3.4 Predict (Unified — Bulk + Composite + Hardness)

- **Default model:** predictions use the chosen default model. Changeable from Predict section dropdown or from Settings
- **Formula input:** text input for chemical composition with:
  - Auto-fix for unicode subscripts
  - Parentheses support
  - Decimal notation support
  - **Strict validation mode (S5):** toggleable from Dataset Management home screen. Enforces:
    - Charset restrictions (A-Z, a-z, 0-9, `.`, `-`, `()`, `{}` only)
    - Bracket balance and nesting hierarchy (`()` inside `{}` only, not vice versa)
    - Element token validation: rejects lowercase-only (`k`, `kananb`), multi-lowercase (`Oo`, `Kaaa`), trailing fragments (`KNaNbO3-ooo`)
    - Defaults to ON. Legacy mode available for relaxed parsing.
- **Composite fields:** additional input controls — only visible/active if the currently selected model was trained on composite features:
  - matrix_type (dropdown: pvdf, p_vdf_trfe, pvdf_hfp, etc.)
  - filler_wt_pct (number input, 0-100)
  - particle_morphology (dropdown: spherical, rod, cube, etc.)
  - particle_size_nm (number input)
  - surface_treatment (dropdown: untreated, silane, plasma, etc.)
  - fabrication_method (dropdown: conventional, hot_press, solvent_cast, etc.)
- **Hardness output:** only shown if current model was trained on vickers_hardness
- **Prediction output display:**
  - d33 (pC/N) with 95% confidence interval + animated gauge bar
  - tc (°C) with 95% CI + animated gauge bar
  - Vickers Hardness (HV kgf/mm²) with **95% CI** gauge (if available) — S5: CI computation added for hardness via ensemble tree std dev
  - Mohs Hardness Scale visualization (if available)
  - **Partial success support (S5):** if one target fails, remaining targets still show results with aggregated error notes
- **Multi-material comparison:** add multiple predictions side-by-side, compare d33/tc/hardness (only properties supported by selected model) across bulk ceramics and composites in a comparison table/chart
- **Batch processing:**
  - Upload CSV → predict for all rows
  - **Multi-target batch (S5):** accepts `model_ids` dict (`{d33: uuid, tc: uuid, vickers_hardness: uuid}`) for per-target model selection. Each formula is predicted against ALL selected models simultaneously. Unselected targets are skipped.
  - Auto-detect bulk vs composite in batch: same logic (filler_wt_pct=0 + matrix_type="none" → bulk ceramic; else composite)
  - **Clean CSV output (S5):** excludes source d33/tc/vickers_hardness columns. Contains: uid, formula, is_composite, predicted values with 95% CI, top_use_case, use_case_score, prediction_status, prediction_notes
  - **Tabular preview (S5):** API returns `results: BatchResultRow[]` allowing frontend to render inline table without re-parsing CSV. Shows per-target columns only for selected models, with CI ranges, score badges, and status indicators. Sticky header, scrollable, error rows highlighted.
- **Usage prediction engine (S5):**
  - 11 research-backed use-case categories: Medical Ultrasound, Implantable Devices, NDT/Industrial Sensors, High-Power Ultrasonics, Automotive Sensors, Aerospace/High-Temp SHM, Energy Harvesting, Sonar/Underwater, Wearable/IoT, Extreme Environment, Precision Actuators/MEMS
  - Rule-based scoring (0–100) with property-specific ideal targets and sigma-based Gaussian fit
  - Confidence tiers: Primary (≥70), Secondary (45–69), Tertiary (30–44)
  - Composite modifiers (+15 wearable, +10 sonar, -20 ultrasonics/NDT)
  - Scientific caution notes generated per material
  - Partial-property scaling (fewer properties = proportionally lower ceiling)
  - `UseCaseCard` UI: tier badges (Highly Recommended/Good Fit/Possible), driving properties as monospace tags, collapsible additional recommendations, caution notes with warning styling
- **Report inclusion:** user can select specific or multiple or all predictions to include in the report PDF. If AI/LLM is configured, include AI insight about predicted use-case based on d33/tc/hardness values. If not configured, skip AI insight with alert
- **Download Report button**
- **Draggable card layout**

### 3.5 Optimization Lab (Structural Analysis AI & Property Optimization)

**Purpose:** This section combines two key capabilities from the 6th semester synopsis Objective 4:

1. **Crystal Structure Analysis** — Use pre-trained AI models to analyze 3D crystal structures as a fast, cost-effective alternative to expensive SEM testing
2. **Multi-Objective Property Optimization** — Find the best balance between competing material properties (d33, tc, hardness) for specific industrial use-cases

**Sub-section A: Crystal Structure Analysis (Feature Extraction)**

- Uses pre-trained universal ML interatomic potentials (M3GNet/CHGNet via Pymatgen/MatGL) as fast surrogate models for structural analysis
- Input: chemical formula → generates approximate 3D perovskite crystal structure (CIF)
- Performs rapid structural relaxation using pre-trained M3GNet/CHGNet (milliseconds vs hours for DFT)
- Extracts graph embeddings (latent vectors from penultimate layer) that encode structural, geometric, and chemical features
- Displays: crystal structure visualization, extracted structural features/descriptors, comparison of structural features across different compositions
- **Key value proposition:** provides structural awareness without expensive electron microscope (SEM) testing or DFT calculations
- **Note:** GNN/CHGNet is deferred to future — for v2.1, this section will use Pymatgen-based structural analysis and physics-based descriptors (tolerance factor, octahedral factor, bond valence, Goldschmidt criteria) as the structural analysis capability. Full GNN transfer learning is marked as a future enhancement

**Sub-section B: Multi-Objective Property Optimization (Pareto Front)**

- **Goal:** Find optimal compositions that balance competing properties — maximizing d33 while maintaining high tc and required hardness
- **Optimization config panel:**
  - Set target property ranges/constraints (min/max for d33, tc, hardness)
  - Define use-case preset profiles:
    - 🔋 Flexible Wearables: high d33 + low hardness + moderate tc
    - ⚡ Industrial Actuators: moderate d33 + high hardness + high tc
    - 🔊 Ultrasonic Transducers: moderate d33 + very high hardness + high tc
    - 🎯 Custom: user-defined ranges
- **NSGA-II multi-objective optimization:** uses trained ML models as surrogate fitness functions to evaluate millions of theoretical compositions in milliseconds
- **Pareto front visualization:** 2D/3D interactive chart showing the optimal trade-off surface between d33, tc, and hardness
- **Solution table:** ranked list of Pareto-optimal compositions with predicted properties and use-case category tag
- **Convergence chart:** optimization progress over generations
- **Use-case mapping results:**
  - Each Pareto-optimal composition gets tagged with its best-fit industrial use-case
  - Color-coded by category (wearables = blue, industrial = orange, ultrasonic = red, etc.)

### 3.6 Interpretability

- **Global Feature Importance (SHAP Beeswarm):** which features most impact predictions, dots color-coded by feature value (low=blue, high=red)
- **Local Prediction Explanation (SHAP Waterfall):** breakdown showing contribution of each feature to a specific single prediction
- **Feature Dependence Plot:** relationship between a specific feature and its SHAP values, revealing non-linear relationships
- **Physics Validation card:** checks if SHAP associations align with expected solid-state physics:
  - Alignment score percentage
  - Violations detected (e.g., "model learned inverse relationship where physics expects positive")
  - Confirmed logic (e.g., "tolerance factor ranks in top 5 by SHAP magnitude — confirms perovskite stability theory")
- **Symbolic Regression (PySR):**
  - Discovers interpretable mathematical equations relating composition features to properties
  - Displays: equation, complexity score, R² fit, comparison to ML model accuracy
  - Parsimony pressure visualization showing accuracy vs complexity trade-off (Pareto front of equations)
- **Info tooltips** on each plot explaining what it represents, its significance for non-ML experts, and how to interpret results
- **Expandable graphs:** full-window view with zoom/pan/move controls (arrows, zoom in/out, reset — similar to GitHub mermaid viewer). Option to hide navigation buttons for clean preview/screenshot
- **KaTeX rendering** for mathematical equations from PySR

### 3.7 Settings

- **System Environment:** dataset count, trained models count, predictions count, DB size
- **Default model selector:** choose which model is used for predictions globally
- **Trained Models Library:** table with target, model name (renameable), algorithm, R², RMSE, created date. Actions: set as default, rename, delete — all changes reflect in DB and propagate across the entire app
- **Pending Elements:** table with element name and supported field(s) (from DB field list). Actions: add new element, edit supported fields, delete - All changes reflect in DB and propagate across the entire app. [PENDING_ELEMENTS], fetch and populate supported fields by looking at the supported elements field in the registry.
- Provide info about list of all the supported elements, properties and capabilites of our app.
- **App Environment Configuration (S9 feature):**
  - UI forms to manage branding: `APP_VERSION`, `APP_NAME`, `APP_LOGO_TEXT`, `APP_TAGLINE`
  - UI to manage developer details: `NEXT_PUBLIC_DEV_NAME`, `NEXT_PUBLIC_DEV_GITHUB`, `NEXT_PUBLIC_DEV_LINKEDIN`
  - Ensures central management of these variables. Changes are written to the `.env` file and picked up by backend/frontend dynamically where applicable, or upon restart.
  - **Priority Order:** The system enforces an industry-standard configuration priority order: (1) Terminal/IDE environment variables (highest priority), (2) `.env` file values, (3) Graceful hardcoded application defaults (lowest priority).
- **API Configuration:**
  - Backend URL, WebSocket URL
  - Database URL
  - API Key (for cloud LLM — OpenAI/Anthropic/Google)
  - LLM Base URL (for local Ollama)
  - LLM Model selection (gpt-4o, claude-sonnet, gemini-flash, local models)
  - Custom model parameters (temperature, max tokens)
  - Changes update `.env` and take effect on API restart
- **Danger Zone:**
  - Purge all models (delete from DB + filesystem) — requires confirmation
  - Clear prediction cache (remove cached results + reports) — requires confirmation
- **Lucide icons** — consistent minimalist style throughout

---

## 4. Data Schema Updates

> [!IMPORTANT]
> **Action Required:** Update `resources/main-datasets/knn_schema_reference.csv` to include proper default/sentinel values for bulk ceramic rows in composite-specific fields:
>
> - `matrix_type` → "none" for bulk ceramics
> - `particle_morphology` → "none" for bulk ceramics
> - `surface_treatment` → "none" for bulk ceramics
> - `particle_size_nm` → NA/null for bulk ceramics
> - `filler_wt_pct` → 0 for bulk ceramics
>
> These defaults must be consistently enforced during dataset upload column mapping and during training data preprocessing.

---

## 5. Central Element Registry & Formula Parsing Strategy

### 5.1 Why a Central Element Registry?

Currently, supported elements are scattered across multiple files:

- `FormulaValidator` has `A_SITE_ELEMENTS` and `B_SITE_ELEMENTS` (18 elements)
- `FeatureEngineer` has `ELEMENTS` list (25 elements, excludes O)
- Physics properties are fetched dynamically from `mendeleev` at runtime

**Problem:** Adding a new element requires changing multiple files. Inconsistencies cause silent parsing/training failures. No single place to see what the platform supports.

**Solution:** A single centralized `ELEMENT_REGISTRY` in `packages/ml-core/piezo_ml/registry/element_registry.py` that:

1. Lists ALL supported elements with pre-computed physics/chemistry properties
2. Is the ONLY source of truth — parser, validator, feature engineer, and API all reference it
3. Is easily expandable — adding a new element = adding one entry to this registry
4. Includes metadata for graceful error messaging when unsupported elements are encountered

### 5.2 Element Registry Design

**Location:** `packages/ml-core/piezo_ml/registry/element_registry.py`

Each element entry contains pre-computed constant properties (hardcoded for consistency and zero-dependency startup).

**Auto-bootstrap mechanism for new elements:**

1. Developer adds a new element symbol to a `PENDING_ELEMENTS` list in the registry
2. On next app startup, a bootstrap script detects pending entries and auto-fetches their properties from `mendeleev`/`pymatgen`
3. Properties are written into the hardcoded registry dict and the element is moved from `PENDING_ELEMENTS` to `ELEMENT_REGISTRY`
4. On subsequent restarts, the element is already hardcoded — no live fetching needed
5. This ensures zero-runtime-dependency while making expansion a one-line addition

Properties per element:

| Property                    | Source             | Why It Matters for Piezoelectrics                                                                      |
| --------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| `atomic_number`             | Periodic table     | Basic identification                                                                                   |
| `symbol`                    | Periodic table     | Key for all lookups                                                                                    |
| `atomic_mass`               | mendeleev          | Weighted average mass affects lattice dynamics and phonon modes                                        |
| `en_pauling`                | mendeleev          | Electronegativity controls bond ionicity/covalency balance — directly affects polarization and d33     |
| `atomic_radius_pm`          | mendeleev          | Atomic radius determines perovskite tolerance factor (t) — controls phase stability and Tc             |
| `ionic_radius_pm`           | Shannon tables     | Ionic radius in coordination environment — critical for A-site/B-site occupancy and lattice distortion |
| `covalent_radius_pm`        | mendeleev          | Covalent bonding character — affects hardness and elastic modulus                                      |
| `vdw_radius_pm`             | mendeleev          | Van der Waals interactions in composite interfacial bonding                                            |
| `melting_point_k`           | mendeleev          | Thermal stability proxy — correlates with Tc for oxide ceramics                                        |
| `boiling_point_k`           | mendeleev          | Volatility during sintering — processing relevance                                                     |
| `electron_affinity_ev`      | mendeleev          | Charge transfer tendency — affects defect chemistry and domain pinning                                 |
| `ionization_energy_ev`      | mendeleev (1st IE) | Energy to remove valence electron — proxy for oxidation state stability                                |
| `valence_electrons`         | mendeleev          | Number of valence electrons — controls bonding character and band structure                            |
| `group`                     | Periodic table     | Periodic group — chemical family behavior                                                              |
| `period`                    | Periodic table     | Periodic period — shell filling affects size and reactivity                                            |
| `block`                     | Periodic table     | s/p/d/f block — determines orbital character of bonding                                                |
| `density_g_cm3`             | mendeleev          | Bulk density — relevant for composite weight fraction calculations                                     |
| `specific_heat_j_gk`        | mendeleev          | Heat capacity — affects sintering behavior and thermal shock resistance                                |
| `thermal_conductivity_w_mk` | mendeleev          | Thermal transport — affects operating temperature limits                                               |
| `bulk_modulus_gpa`          | Literature/DFT     | Resistance to compression — directly correlates with mechanical hardness (Pugh's ratio)                |
| `shear_modulus_gpa`         | Literature/DFT     | Resistance to shear — Pugh's G/B ratio predicts ductile/brittle behavior                               |
| `youngs_modulus_gpa`        | Literature/DFT     | Stiffness — affects elastic compliance and d33 via d = eS (piezo = charge × compliance)                |
| `poisson_ratio`             | Literature/DFT     | Lateral vs axial strain — distinguishes brittle ceramics from ductile metals                           |
| `polarizability_a3`         | CRC Handbook       | Electronic polarizability — high polarizability → easier dipole formation → higher d33                 |
| `oxidation_states`          | mendeleev          | Common oxidation states — determines charge balance in perovskite ABO3                                 |
| `coordination_number`       | Crystal chemistry  | Preferred coordination — A-site (12-fold) vs B-site (6-fold) determination                             |
| `perovskite_site`           | Domain knowledge   | "A", "B", "O", or "dopant" — helps validate stoichiometry                                              |
| `is_rare_earth`             | Periodic table     | Flag for lanthanide dopants (La, Nd, Pr, Sm, Eu, Gd, Ho)                                               |

**Supported elements (v2.1 — 30 elements + O):**

| Category                   | Elements                                  | Count  |
| -------------------------- | ----------------------------------------- | ------ |
| A-site (alkali/alkaline)   | K, Na, Li, Ba, Ca, Sr, Ag                 | 7      |
| A-site (bismuth family)    | Bi, Pb                                    | 2      |
| B-site (transition metals) | Nb, Ta, Ti, Zr, Hf, Sb, W, Mo, Sn, Sc, Fe | 11     |
| Dopants / modifiers        | Cu, Mn, Al, Mg, Zn                        | 5      |
| Rare earth dopants         | La, Nd, Pr, Sm, Eu, Gd, Ho                | 7      |
| Anion                      | O                                         | 1      |
| **Total**                  |                                           | **33** |

> [!TIP]
> **Expanding support:** To add a new element (e.g., Ce, Y, Cr), add its symbol to `PENDING_ELEMENTS`. On next startup, the auto-bootstrap fetches and hardcodes all properties. The parser, validator, feature engineer, and API will automatically recognize it. No other file changes needed.

### 5.3 Formula Parsing Strategy

**Analysis of dataset formulas reveals these patterns:**

| Pattern                                | Example                                                                               | Complexity          |
| -------------------------------------- | ------------------------------------------------------------------------------------- | ------------------- |
| Simple perovskite                      | `KNbO3`                                                                               | Basic               |
| Fractional site occupancy              | `K0.5Na0.5NbO3`                                                                       | Standard            |
| Nested parentheses                     | `(K0.44Na0.52Li0.04)(Nb0.84Ta0.10Sb0.06)O3`                                           | Medium              |
| Multi-phase solid solution             | `0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5Na0.5ZrO3`                                | Complex             |
| 3+ phase mixtures                      | `0.964K0.4Na0.6Nb0.955Sb0.045O3-0.006BiFeO3-0.03Bi0.5Na0.5ZrO3`                       | Complex             |
| Deep nesting with multiplier           | `0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5(Na0.82K0.18)0.5ZrO3`                     | Very Complex        |
| 4-phase with dopant                    | `0.944K0.48Na0.52Nb0.95Sb0.05O3-0.04Bi0.5(Na0.82K0.18)0.5ZrO3-0.016AgSbO3-0.004Fe2O3` | Very Complex        |
| Unicode subscripts (master dataset)    | `Na₀.₅₃₅K₀.₄₈NbO₃`                                                                    | Encoding fix needed |
| Garbled dashes (XLSX→CSV)              | `â€"`, `âˆ'`, `—`, `–`, `−`                                                           | Encoding fix needed |
| Full-width brackets (XLSX→CSV)         | `ï¼ˆ`, `ï¼‰`, `（`, `）`                                                              | Encoding fix needed |
| Descriptive text in formula            | `K0.5Na0.5NbO3(Zn-Sn doped)`                                                          | Needs stripping     |
| Composite polymer name (NOT a formula) | `pvdf`, `pvdf_trfe`                                                                   | Skip — not parsed   |

**chemparse verdict:** chemparse handles simple and parenthesized formulas correctly but:

- ❌ Does NOT handle multi-phase dash-separated formulas (`A-B-C`) — it treats the whole string as one formula
- ❌ Does NOT handle leading multipliers like `0.96(...)` properly in all cases
- ❌ No chemical intelligence — can't validate charge balance or perovskite rules
- ✅ Good for individual single-phase formula parsing within parentheses

**Strategy:** Keep chemparse as the low-level single-phase parser (it's fast and handles parentheses/fractions well). Our custom `FormulaParser` already wraps it correctly — it splits multi-phase formulas at top-level dashes, extracts leading multipliers, then delegates each phase to chemparse. This architecture is sound and should be retained.

**Additional improvements for v2.1:**

1. Pre-normalize unicode subscripts (₀₁₂₃₄₅₆₇₈₉ → 0-9) and garbled dashes BEFORE parsing
2. Strip descriptive text in parentheses (e.g., "(Zn-Sn doped)" → removed)
3. Validate ALL parsed elements against the Central Element Registry
4. Return structured result with: `{elements: Dict, warnings: List, unsupported: List, is_valid: bool}`
5. **Formula validation at dataset upload:** run formula parsing during the Review Issues step. Unsupported elements and parse errors appear as issues with inline edit. **Auto-fix** only for safe operations (unicode normalization, whitespace stripping). Ambiguous cases require manual user fix — do NOT auto-fix anything that could alter stoichiometry or break training process.

### 5.4 Feature Engineering from Parsed Formulas

After parsing a formula like `0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5Na0.5ZrO3`:

**Step 1 — Stoichiometric calculation:**

```
K:  0.96×0.48 = 0.4608
Na: 0.96×0.52 + 0.04×0.5 = 0.5192
Nb: 0.96×0.95 = 0.912
Sb: 0.96×0.05 = 0.048
O:  0.96×3 + 0.04×3 = 3.0
Bi: 0.04×0.5 = 0.02
Zr: 0.04×1 = 0.04
```

**Step 2 — Elemental mole fractions (excluding O):**
Total non-O atoms = 0.4608 + 0.5192 + 0.912 + 0.048 + 0.02 + 0.04 = 2.0
Each element's fraction = stoich_count / total_non_O

**Step 3 — Physics descriptors (weighted by mole fraction):**
For each property in Element Registry, compute: `weighted_mean`, `weighted_variance`
Plus: tolerance factor, octahedral factor

**Step 4 — Composite vector (8-dim, zero-padded for bulk):**
Appended from composite fields in the dataset

---

## 6. Training & Prediction Artifact Storage

### 6.1 Parsed Dataset Storage (for Manual Verification)

**Purpose:** Every time a dataset is preprocessed for training, save the parsed elemental compositions for corresponding formulas in the same row order alongside the source dataset. Each row includes the **uid** from the source dataset so users can verify which original row maps to which parsed composition, even if rows were dropped during preprocessing.

**Location:** `resources/training-artifacts/`

**Structure:**

```
resources/training-artifacts/
├── dataset_<dataset_id>_<YYYYMMDD_HHMMSS>/
│   ├── source_with_uid.csv            # Original dataset + uid column (sequential from 1)
│   ├── parsed_compositions.csv        # Parsed elements per formula, same row order + uid
│   ├── feature_vectors.csv            # Full feature vectors used for training
│   └── preprocessing_log.txt          # Log of all preprocessing steps
```

**`source_with_uid.csv`** — Original dataset with added `uid` column (sequential from 1, based on original upload order):

```csv
uid,formula,d33,tc,...
1,K0.5Na0.5NbO3,151.0,420,...
2,(K0.44Na0.52Li0.04)(Nb0.84Ta0.10Sb0.06)O3,416.0,253,...
```

**`parsed_compositions.csv`** — Parsed elements in exact same order, with matching uid:

```csv
uid,formula,K,Na,Li,Nb,Ta,Sb,O,Bi,Zr,...,parse_status,parse_warnings
1,K0.5Na0.5NbO3,0.5,0.5,0,1,0,0,3,0,0,...,success,
2,(K0.44Na0.52Li0.04)(Nb0.84Ta0.10Sb0.06)O3,0.44,0.52,0.04,0.84,0.10,0.06,3,0,0,...,success,
```

This allows the user to:

- Open both CSVs side-by-side
- Download the parsed csv and the src csv.
- Verify formula in row N with unique uid of source matches parsed elements with that uid row of parsed file. Match elements in ascending uid order (alphabetical order of elements). Unmatched uid rows comes in same order with a corresponding blank field/row in either src or parsed dataset which is missing that uid. This will help in better comparison of main src dataset vs its corresponding correctly matched parsed dataset.
- Spot any parsing errors (wrong element assignment, incorrect stoichiometry calculation/multiplication, missing elements)

**Parsed Dataset Comparison UI** (sub-section inside Dataset Upload & Management):

- Side-by-side tabular view of source vs parsed datasets, uid-mapped
- Search by formula or uid
- Highlight mismatches or rows with parse warnings
- Same crud operation for cells/rows/columns as Dataset Explorer.
- **True Source vs Parsed comparison (S2 hardening):** persist `materials.source_row` (source snapshot) and `materials.parsed_row` (normalized/validated snapshot) so the comparison view shows real differences (e.g., normalized formula) instead of duplicating the same payload.
- **Comparison search rule (S2 hardening):** row filtering uses the intersection of **Search In** fields and currently visible columns, preventing matches in hidden columns that users cannot inspect.

### 6.2 Trained Model Artifact Storage

**Purpose:** Save trained models with consistent naming that maps to the dataset and configuration used.

**Location:** `resources/trained-models/`

**Naming convention:** `model_<target>_<algorithm>_<YYYYMMDD_HHMMSS>.joblib`

**Structure:**

```
resources/trained-models/
├── model_d33_xgboost_20260507_143022.joblib
├── model_tc_randomforest_20260507_143022.joblib
├── model_hardness_gbr_20260507_143022.joblib
├── metadata_20260507_143022.json
```

**`metadata_<timestamp>.json`** — [So verify if the below json schema contains/links the parsed dataset too if not then add it because i added the parsed dataset in this description later in this plan] Links model to its source dataset, parsed dataset used for training and training config. All references use **UUID identifiers** (never change even if model/dataset is renamed), ensuring consistent DB integrity:

```json
{
  "training_id": "uuid",
  "timestamp": "2026-05-07T14:30:22",
  "dataset_id": "uuid",
  "dataset_name": "knn_bulk_ceramic_dataset",
  "source_artifact_dir": "resources/training-artifacts/dataset_xxx_20260507_143022/",
  "targets": ["d33", "tc"],
  "algorithms": {"d33": "xgboost", "tc": "randomforest"},
  "hyperparameters": {"d33": {"n_estimators": 100, ...}, "tc": {...}},
  "metrics": {"d33": {"r2": 0.92, "rmse": 45.2}, "tc": {"r2": 0.88, "rmse": 22.1}},
  "feature_version": "v4",
  "feature_dim": 45,
  "n_train_samples": 28,
  "n_test_samples": 8,
  "supported_elements": ["K", "Na", "Nb", "O", ...],
  "model_files": ["model_d33_xgboost_20260507_143022.joblib", ...]
}
```

---

## 7. Element Validation & Error Handling (Train + Predict)

### During Training (Dataset Upload + Preprocessing)

1. Log initial dataset dimensions: "Starting with X rows × Y columns"
2. Parse each formula in the dataset against the Element Registry
3. For rows with unsupported elements:
   - **Warning in terminal logs:** `⚠️ Row 15 (uid=15): Formula "K0.5Na0.5NbO3(Zn-Sn doped)" contains unsupported elements: [Zn, Sn]. Marked for review.`
   - **UI data quality report:** Show in Review Issues step with reason: "Contains unsupported elements: Zn, Sn"
   - **Option:** Rows with issues are flagged (not auto-skipped) — user sees them in Review Issues and can: (a) edit the formula inline to fix it, (b) explicitly skip/delete the row, or (c) skip all flagged rows at once. This gives the user control before any data is discarded. And this option should be selected at the start of the training process so that if user want to skip/delete/drop all flagged rows then he does not have to wait for that point to come and he can do other tasks in between. Only if he select to edit or skip/delete based on each row condition then only he will have to wait ad see the complete process, else in case of skip/drop all it should run in background and user can do other tasks in between. Prove short reason for flagged issue to and what is expected.
4. After all rows processed, show summary: "Processed 35/37 formulas successfully. 2 rows flagged with issues." Include final dimensions: "Final dataset: X rows × Y columns"

### During Prediction (Single + Batch)

1. Parse input formula against Element Registry
2. If unsupported elements found:
   - **Do NOT predict** — return friendly error
   - **UI message:** "⚠️ The formula contains elements not yet supported by our platform: **[Zn, Sn]**. Piezo.AI currently supports 33 elements commonly found in perovskite piezoelectrics. We are actively working to expand element support in future updates."
   - **Note:** For batch prediction via CSV upload, unsupported element issues are ideally caught and fixed during the Dataset Upload step (which has full CRUD editing). The prediction section shows a read-only error for individual formula input. Ensure that the formulas are not decomposed into individual element by the parser it only upload and check the validity of the dataset, because in prediction we should input complete formula of cermaics and composites property and predict its d33, tc, hardness(if applicable and supported by model). So think logically for this implementation.
   - **Show supported elements list** in a collapsible section
3. If formula parse fails entirely (bad syntax):
   - **UI message:** "❌ Could not parse the chemical formula. Please check for typos, unmatched brackets, or invalid characters. Examples of valid formulas: K0.5Na0.5NbO3, (K0.44Na0.52Li0.04)(Nb0.84Ta0.10Sb0.06)O3"

### During Batch Prediction

- For each row, flag unsupported/unparseable formulas in the result CSV:
  - Add `prediction_status` column: "success", "unsupported_elements", "parse_error"
  - Add `prediction_notes` column: specific error message
  - Successfully predicted rows get their d33_predicted, tc_predicted values
  - Failed rows get blank prediction columns + error in notes
