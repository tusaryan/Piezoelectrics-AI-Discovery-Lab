# Features Guide

> Detailed reference for every platform feature: Dataset management, Prediction engine, SHAP Interpretability, NSGA-II Optimization, Use-Case Mapping, Symbolic Regression, PDF Reports, and Settings.

**← [Back to Developer Guide](DEVELOPER_GUIDE.md)**

---

## Table of Contents

- [Dataset Management](#dataset-management)
- [Prediction Engine](#prediction-engine)
- [Interpretability (SHAP)](#interpretability-shap)
- [Physics Validation](#physics-validation)
- [Symbolic Regression (PySR)](#symbolic-regression-pysr)
- [Optimization Lab (NSGA-II)](#optimization-lab-nsga-ii)
- [Use-Case Mapping Engine](#use-case-mapping-engine)
- [PDF Report Generation](#pdf-report-generation)
- [Settings Page](#settings-page)

---

## Dataset Management

**Page:** Dataset | **Backend:** `apps/api/app/modules/dataset/` | **Store:** `datasetStore.ts`

### Upload Wizard (4-Step)

The dataset upload follows a guided wizard flow:

#### Step 1: Upload
- Accepts `.csv` files only
- File is parsed server-side, column names + sample rows returned for preview
- Maximum file size: configurable (default: unlimited)

#### Step 2: Map Columns
- The system presents all CSV columns and asks you to map them to the internal schema
- **Required mapping:** `formula` (chemical composition column)
- **Optional mappings:** `d33`, `tc`, `vickers_hardness`, `qm`, `kp`, and all composite fields
- Auto-detection: If your CSV has columns named `d33`, `formula`, etc., they're pre-mapped
- Unmapped columns are silently ignored

#### Step 3: Review Issues
- After mapping, the system validates every row:
  - **Formula validation:** Each formula is parsed through `FormulaParser`. Invalid/unparsable formulas are flagged.
  - **Column quality metrics:** Per-column validity percentage, missing count, and issue count
  - **Issue types:** Missing values, invalid formula tokens, out-of-range numerics
- You can select columns for bulk clearing (remove values across all rows)
- 3 resolution options: Fix manually, drop rows with issues, or proceed with issues

#### Step 4: Explore
- Interactive data table (TanStack Table) with sorting, filtering, and pagination
- Full dataset preview with all mapped columns

### Dataset Table Features

- **View:** Eye icon — opens full row preview
- **Download:** Export dataset as CSV
- **Delete:** Remove dataset + cascade delete all materials, training jobs, and models from this dataset
- **Status badges:** `Ready` (green), `Processing` (yellow), `Error` (red)

---

## Prediction Engine

**Page:** Predict | **Backend:** `apps/api/app/modules/prediction/` | **ML:** `packages/ml-core/piezo_ml/models/inference_engine.py`

### Single Prediction

1. Type or paste a chemical formula into the input field
2. The `FormulaValidationInput` component provides live validation:
   - Green checkmark: valid formula, all elements supported
   - Red cross: parse error or unsupported elements (with specific feedback)
3. Click "Predict" — the formula goes through:
   - `FormulaParser.parse()` → element extraction
   - `FeatureEngineer.engineer_row()` → feature vector
   - `InferenceEngine.predict_single()` → model inference
4. Results displayed as cards:
   - **d₃₃** (pC/N) with 95% CI
   - **Tc** (°C) with 95% CI
   - **Vickers Hardness** (HV) with 95% CI
   - **Suggested Use Case** — top recommendation from the Use-Case Mapper

### Confidence Intervals

For tree-based ensemble models (Random Forest, XGBoost, LightGBM, Gradient Boosting, Stacking):
- CI computed from individual tree predictions: `prediction ± 1.96 × std(tree_predictions)`
- This gives a true 95% confidence interval based on model variance

For non-ensemble models (SVR, Decision Tree, ANN):
- Fallback: `prediction ± 10% × |prediction|`

### Batch Prediction

1. Upload a CSV with a `formula` column
2. The system processes all rows in parallel
3. Results available as a downloadable CSV with predicted d₃₃, Tc, Hardness, and CIs for each formula
4. Success/error counts shown in the batch summary card

### Composite Material Prediction

When `ENABLE_COMPOSITE_MODULE=true`, the prediction form expands to include:
- Matrix type (PVDF, Epoxy, etc.)
- Filler weight percentage
- Particle size/morphology
- Sintering method/temperature
- Surface treatment

These composite parameters are encoded via `composite_encoder.py` and appended to the feature vector before inference.

---

## Interpretability (SHAP)

**Page:** Interpretability | **Backend:** `apps/api/app/modules/interpret/` | **ML:** `packages/ml-core/piezo_ml/evaluation/shap_analyzer.py`

### Three SHAP Modes

#### 1. Beeswarm Plot (Global)
- Shows feature importance across **all training samples**
- Each dot = one sample. Position on X-axis = SHAP value (impact on prediction). Color = feature value (high/low).
- Reveals which features have the most predictive power across the dataset
- Example insight: "Higher `electronegativity_weighted_var` pushes d₃₃ predictions upward" — meaning compositions with diverse electronegativities tend to have higher piezoelectric coefficients

#### 2. Waterfall Plot (Local — Per-Sample)
- Decomposes a **single prediction** into feature contributions
- Shows base value → final prediction path
- Navigate across samples (1/50 sample selector)
- Example: For BaTiO₃, the waterfall might show `frac_Ba = +15.3`, `ionic_radius_pm_weighted_mean = +8.2`, meaning Ba content and ionic radius strongly push the prediction up

#### 3. Feature Dependence Plot
- Shows how **one specific feature** affects predictions across all samples
- Scatter plot: X = feature value, Y = SHAP value
- Color = interaction feature (auto-selected for maximum interaction)
- Reveals non-linear relationships and feature interactions

### SHAP Computation & Caching

- SHAP analysis runs as a **background task** (not blocking the request)
- Frontend polls for completion via `GET /api/v1/interpret/shap/{task_id}/status`
- Results are cached to disk in `resources/shap-cache/` with model+dataset composite keys
- Cache invalidated when the model is retrained or dataset changes
- Uses `shap.TreeExplainer` for tree-based models and `shap.KernelExplainer` for others

### Performance

- 50 samples × 91 features takes ~3–15 seconds (depending on algorithm)
- Progress is reported via polling response
- Background task architecture prevents Next.js proxy timeouts (the v2.1.0 fix)

---

## Physics Validation

**File:** `packages/ml-core/piezo_ml/evaluation/physics_validator.py`

After SHAP analysis, the Physics Validator checks whether the model's learned feature importances align with established solid-state physics:

### Validation Rules

| Rule | Physics Basis | Check |
|------|--------------|-------|
| Ionic radius matters | Perovskite stability is governed by tolerance factor (r_A + r_O)/(√2(r_B + r_O)) | `ionic_radius_pm` should be in top-15 SHAP features |
| Electronegativity matters | Bonding character (ionic vs covalent) drives polarization | `electronegativity` features should show non-zero SHAP |
| Tolerance factor present | Goldschmidt tolerance factor is the primary perovskite stability predictor | `tolerance_factor` should have measurable SHAP impact |
| Compositional balance | Element fractions should collectively contribute | At least one `frac_*` feature should be significant |

### Output

- **Percentage score** (e.g., "100% — Strong Alignment", "75% — Good Alignment")
- Per-rule pass/fail with explanations
- Displayed in the Interpretability page as a circular progress indicator

---

## Symbolic Regression (PySR)

**File:** `packages/ml-core/piezo_ml/symbolic_regression/`

PySR uses Julia-based genetic programming to discover compact mathematical equations that approximate the ML model's learned relationships.

### How It Works

1. SHAP feature importances identify the top-N most influential features
2. PySR searches for equations of the form: `d₃₃ ≈ f(feature_1, feature_2, ...)`
3. The search runs for 1–3 minutes (background task with polling)
4. Results are ranked by complexity vs. accuracy (Pareto front of equations)
5. Equations are rendered in **KaTeX** in the frontend for publication-ready display

### Requirements

- **Julia runtime** must be installed on the system
- PySR installs its Julia dependencies on first run (~500MB)
- On macOS, Julia may require manual permission grants (see [Troubleshooting](TROUBLESHOOTING.md))

### Example Output

```
Complexity 3:  d₃₃ ≈ 142.5 × electronegativity_weighted_var
Complexity 7:  d₃₃ ≈ 95.2 × frac_Bi + 142.5 × electronegativity_weighted_var - 12.3
Complexity 12: d₃₃ ≈ 95.2 × frac_Bi × (1 + 0.42 × tolerance_factor) + 142.5 × electronegativity_weighted_var
```

---

## Optimization Lab (NSGA-II)

**Page:** Optimization Lab | **Backend:** `apps/api/app/modules/optimization/` | **ML:** `packages/ml-core/piezo_ml/optimization/nsga2_optimizer.py`

### What It Solves

Most piezoelectric design involves competing objectives:
- **High d₃₃** (strong piezo response) usually means **low Tc** (thermally fragile)
- **High Tc** (thermally stable) usually means **low d₃₃** (weak response)
- **High Hardness** (mechanically tough) often conflicts with both

NSGA-II (Non-dominated Sorting Genetic Algorithm II) finds the **Pareto front** — the set of compositions where you can't improve one objective without worsening another.

### Configuration

#### Surrogate Models
For each target being optimized, you select a trained model to act as the fitness function:
- d₃₃ → e.g., `knn_stacking_d33_20260518` (R²=0.827)
- Tc → e.g., `all_stacking_tc_20260518` (R²=0.586)
- Hardness → e.g., `stacking_vickers_hardness_20260613` (R²=0.923)

The optimizer uses these models to evaluate candidate compositions without lab synthesis.

#### Use-Case Presets

| Preset | d₃₃ Range | Tc Range | Hardness Range | Scenario |
|--------|-----------|----------|----------------|----------|
| **Flexible Wearables** | High (≥200) | Low-moderate (80–200°C) | Low (≤400 HV) | Skin-conformable health sensors |
| **Industrial Actuators** | Moderate (100–500) | High (≥300°C) | High (≥600 HV) | Motor-adjacent precision actuation |
| **Ultrasonic Transducers** | Moderate (150–600) | Very high (≥350°C) | Very high (≥700 HV) | Continuous-duty ultrasonic cleaning/welding |
| **Custom** | User-defined | User-defined | User-defined | Any target range |

#### Target Objectives
For each target, configure:
- **Direction:** Maximize or minimize
- **Weight:** 0.0–1.0 (relative importance)
- **Range:** min–max bounds for feasibility

### NSGA-II Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population size | 100 | Candidate compositions per generation |
| Generations | 50 | Evolutionary iterations |
| Crossover probability | 0.9 | SBX crossover rate |
| Mutation probability | 1/n_var | Per-variable polynomial mutation |
| n_offspring | 100 | Children per generation |

### Results

1. **Pareto Front Chart** — 2D/3D scatter plot with selectable axes (d₃₃ vs Tc, Hardness vs d₃₃, etc.)
   - Each point = one Pareto-optimal composition
   - Points are color-coded by use-case tag (Ultrasonic Transducer, Sonar/Underwater, etc.)
2. **Solution Table** — Ranked list of all Pareto-optimal compositions with predicted properties
3. **Convergence Chart** — Average objective value across generations (lower = converging)
4. **Timing** — Total optimization time (typically 30–60 seconds for 50 generations)

### Caching

Results are cached in `resources/optimization-cache/` with keys derived from:
- Selected surrogate models
- Target ranges and weights
- Population size and generations

Cache is invalidated when surrogate models are retrained.

### Structure Analysis Tab

A second tab in the Optimization Lab provides crystal structure analysis:
- Tolerance factor distribution across Pareto solutions
- Octahedral factor statistics
- Predicted crystal symmetry classification

---

## Use-Case Mapping Engine

**File:** `packages/ml-core/piezo_ml/models/use_case_mapper.py` (467 lines)

### 11 Application Categories

| # | Category | Icon | Primary Driver | d₃₃ Threshold | Tc Threshold | Hardness Threshold |
|---|----------|------|---------------|----------------|--------------|-------------------|
| 1 | Medical Ultrasound Imaging | 🏥 | d₃₃ | ≥500 (40pts), ≥200 (20pts) | 100–350°C (25pts) | <600 (15pts) |
| 2 | Implantable Biomedical Devices | 🫀 | d₃₃ | ≥1000 (45pts), ≥500 (25pts) | 100–250°C (25pts) | <400 (20pts) |
| 3 | NDT / Industrial Sensors | 🏭 | Tc | 50–500 (30pts) | 200–500°C (35pts) | 500–1000 (25pts) |
| 4 | High-Power Ultrasonics | ⚡ | Hardness | ≥200 (25pts) | ≥300°C (30pts) | ≥700 (35pts) |
| 5 | Automotive Sensors | 🚗 | Tc | 50–500 (25pts) | ≥350°C (40pts) | ≥600 (25pts) |
| 6 | Aerospace / High-Temp SHM | ✈️ | Tc | ≥100 (20pts) | ≥450°C (45pts) | ≥800 (25pts) |
| 7 | Energy Harvesting | 🔋 | d₃₃ | ≥150 (40pts) | ≥200°C (25pts) | any (15pts) |
| 8 | Sonar / Underwater Acoustics | 🌊 | Hardness | 50–600 (30pts) | ≥250°C (25pts) | ≥700 (35pts) |
| 9 | Wearable / IoT Sensors | ⌚ | Hardness (inv) | ≥100 (30pts) | 100–250°C (20pts) | <400 (35pts) |
| 10 | Extreme Environment Sensing | ☢️ | Tc | 10–100 (20pts) | ≥600°C (55pts) | ≥900 (20pts) |
| 11 | Precision Actuators / MEMS | 🔬 | d₃₃ | ≥300 (40pts) | ≥200°C (25pts) | 400–900 (25pts) |

### Scoring Mechanism

Each use case has independent scoring logic (0–100 points):
1. **Property-specific thresholds** contribute points based on how well the material fits
2. **Composite modifiers** boost wearable/biomedical scores (+15) and penalize high-power ultrasonics (-20)
3. **Partial-property scaling:** If only 2 of 3 properties are available, scores are scaled up proportionally to prevent unfair penalization

### Confidence Tiers

| Tier | Score Range | Label | UI Treatment |
|------|------------|-------|-------------|
| **Primary** | ≥70 | "Highly Recommended" | Green badge, prominent display |
| **Secondary** | 45–69 | "Good Fit" | Yellow badge, secondary display |
| **Tertiary** | 30–44 | "Possible Application" | Gray badge, collapsed section |
| Excluded | <30 | — | Not shown |

### Caution Notes

The engine generates context-specific scientific warnings:
- `Tc < 200°C`: "Low Curie temperature limits deployment — max operating temp ≈{Tc/2}°C"
- `d₃₃ < 50`: "Low coefficient restricts use to sensing; unsuitable for actuation"
- `Hardness < 300`: "Indicates flexible/polymer-class material; unsuitable for rigid environments"
- `d₃₃ > 1000 + Tc < 150`: "PMN-PT class — exceptional sensitivity but requires thermal management"

---

## PDF Report Generation

**File:** `packages/ml-core/piezo_ml/reporting/`

Generate downloadable PDF reports from the Dashboard containing:

1. **Dataset Summary** — row counts, column statistics, missing value analysis
2. **Training Results** — per-target R², RMSE, algorithm used, convergence data
3. **Model Comparison Table** — all trained models with performance metrics
4. **Prediction Summary** — recent predictions with properties and CIs
5. **Optimization Results** — Pareto front solutions (if optimization was run)
6. **Charts** — Embedded Matplotlib-generated charts (convergence, scatter)

Generated using **ReportLab** with custom styling. Output saved to `resources/reports/`.

If an LLM is configured (see [Setup Guide → LLM Configuration](SETUP_GUIDE.md#llm-configuration-optional)), AI-generated insights are appended to each section (e.g., "The stacking ensemble's high R² on d₃₃ suggests strong non-linear interactions between compositional features...").

---

## Settings Page

**Page:** Settings | **Backend:** `apps/api/app/modules/settings/` | **Store:** `settingsStore.ts`

### Sections

#### 1. Element Registry
- View, edit, and add elements to the Central Element Registry
- Modify any of the 27 properties per element
- Add new elements (auto-bootstrap from mendeleev/pymatgen or manual entry)
- Change perovskite site classification (A/B/X/None)

#### 2. Field Schema
- Configure which fields are enabled/disabled
- Manage valid options for categorical fields (ceramic_type, fabrication_method, etc.)
- Add custom categorical values

#### 3. ML Limits
- Configure training limits (max epochs, timeout, etc.)
- View model storage usage

#### 4. API Configuration
- View/test backend connection
- Health check status

#### 5. Theme
- Three themes: **Light**, **Dark** (default), **Night** (pure black)
- Managed by `next-themes`
- Persists across sessions via localStorage

---

**← [ML Pipeline](ML_PIPELINE.md)** | **Next: [Interface Gallery →](INTERFACE_GALLERY.md)**
