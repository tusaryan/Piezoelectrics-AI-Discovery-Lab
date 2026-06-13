# ML Pipeline Reference

> Deep dive into the Central Element Registry, formula parsing, feature engineering, all 8 ML algorithms, hyperparameter reference, training orchestration, and data preprocessing.

**← [Back to Developer Guide](DEVELOPER_GUIDE.md)**

---

## Table of Contents

- [Central Element Registry](#central-element-registry)
- [Formula Parsing](#formula-parsing)
- [Feature Engineering](#feature-engineering)
- [Supported Algorithms (8)](#supported-algorithms)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Training Orchestration](#training-orchestration)
- [Data Preprocessing & Missing Values](#data-preprocessing--missing-values)
- [Auto-Tune (Optuna)](#auto-tune-optuna)
- [Model Registry & Versioning](#model-registry--versioning)

---

## Central Element Registry

**File:** `packages/ml-core/piezo_ml/registry/element_registry.py`
**Data:** `packages/ml-core/piezo_ml/registry/element_registry_data.json`

The Element Registry is the scientific backbone of the platform. It stores 42 elements with 27 physics/chemistry properties each, totaling **1,134 property values** bootstrapped from `mendeleev` and `pymatgen` databases.

### Supported Elements (42)

**A-site (perovskite):** Ba, Bi, Ca, K, La, Li, Na, Pb, Sr
**B-site (perovskite):** Co, Cr, Cu, Fe, Ga, Ge, Hf, In, Ir, Mg, Mn, Mo, Nb, Ni, Ru, Sb, Sc, Sn, Ta, Ti, V, W, Y, Zn, Zr
**Framework/Anion:** F, N, O, S
**Dopant/Modifier:** Ag, Al, Ce, Si

### Properties Per Element (27)

| # | Property Key | Unit | Source | Description |
|---|-------------|------|--------|-------------|
| 1 | `atomic_number` | — | mendeleev | Atomic number (Z) |
| 2 | `atomic_mass` | amu | mendeleev | Relative atomic mass |
| 3 | `atomic_radius_pm` | pm | mendeleev | Empirical atomic radius |
| 4 | `ionic_radius_pm` | pm | pymatgen/manual | Shannon ionic radius (common oxidation state) |
| 5 | `covalent_radius_pm` | pm | mendeleev | Covalent bond radius |
| 6 | `vdw_radius_pm` | pm | mendeleev | Van der Waals radius |
| 7 | `electronegativity` | Pauling | mendeleev | Pauling electronegativity scale |
| 8 | `electron_affinity_ev` | eV | mendeleev | First electron affinity |
| 9 | `ionization_energy_ev` | eV | mendeleev | First ionization energy |
| 10 | `polarizability` | Å³ | mendeleev | Dipole polarizability |
| 11 | `melting_point_k` | K | mendeleev | Melting point |
| 12 | `boiling_point_k` | K | mendeleev | Boiling point |
| 13 | `density_g_cm3` | g/cm³ | mendeleev | Solid-state density |
| 14 | `molar_volume_cm3` | cm³/mol | mendeleev | Molar volume |
| 15 | `thermal_conductivity_w_mk` | W/mK | mendeleev | Thermal conductivity |
| 16 | `specific_heat_j_gk` | J/gK | mendeleev | Specific heat capacity |
| 17 | `valence_electrons` | — | mendeleev | Number of valence electrons |
| 18 | `oxidation_states` | — | mendeleev | Common oxidation states (list) |
| 19 | `block` | — | mendeleev | s/p/d/f block classification |
| 20 | `group` | — | mendeleev | Periodic table group |
| 21 | `period` | — | mendeleev | Periodic table period |
| 22 | `is_rare_earth` | bool | derived | Lanthanide/Actinide flag |
| 23 | `perovskite_site` | A/B/X/— | curated | Preferred perovskite lattice site |
| 24 | `en_allen` | — | mendeleev | Allen electronegativity scale |
| 25 | `is_radioactive` | bool | mendeleev | Radioactivity flag |
| 26 | `crystal_structure` | — | mendeleev | Room-temperature crystal structure |
| 27 | `symbol` | — | — | Element symbol (key) |

### Adding New Elements

New elements can be added via the **Settings → Element Registry** page in the UI, or programmatically:

```python
from piezo_ml.registry.bootstrap_element import bootstrap_element

# Auto-fetches from mendeleev + pymatgen, assigns perovskite site
new_entry = bootstrap_element("Nd")
# Then add to element_registry_data.json via the Settings API
```

The UI also provides manual editing of all 27 properties and perovskite site classification.

---

## Formula Parsing

**File:** `packages/ml-core/piezo_ml/parsers/formula_parser.py`

The parser handles everything from simple binary compounds to complex multi-phase solid solutions with parenthetical nesting.

### Supported Formula Formats

| Format | Example | Notes |
|--------|---------|-------|
| Simple binary | `BaTiO3` | Direct stoichiometry |
| Doped perovskite | `Ba0.85Ca0.15Ti0.9Zr0.1O3` | Fractional subscripts |
| Parenthetical | `(K0.5Na0.5)NbO3` | Automatic distribution |
| Multi-phase | `0.96KNN–0.04BNZ` | Phase-fraction splitting |
| Complex solid solution | `0.96(K₀.₄₈Na₀.₅₂)(Nb₀.₉₅Sb₀.₀₅)O₃–0.04Bi₀.₅Na₀.₅ZrO₃` | Full complexity |
| Unicode subscripts | `K₀.₅Na₀.₅NbO₃` | Auto-converted to ASCII |
| With dopants | `BaTiO3 + 0.5wt% MnO2` | Detected (strict mode flags) |

### Parsing Pipeline

1. **Pre-normalization:** Strip whitespace, convert Unicode subscripts (`₀₁₂₃₄₅₆₇₈₉`) to ASCII, standardize delimiters (`–`, `—`, `-` → normalized dash)
2. **Phase splitting:** Detect `0.96X–0.04Y` patterns, extract multipliers
3. **Parenthesis resolution:** DFS scan expands nested groups: `(K0.5Na0.5)` → `K0.5Na0.5`
4. **Element extraction:** `chemparse` tokenizes each phase into `{symbol: count}` pairs
5. **Multiplier application:** Phase fractions multiply element counts: `0.96 × K0.48 → K0.4608`
6. **Cross-phase summation:** Element counts merged across all phases
7. **Validation:** All elements checked against the Central Element Registry

### Strict Mode vs Permissive Mode

- **Strict mode** (default for uploads): Rejects tokens like `wt%`, `mol%`, `+` operators, and unknown element symbols. Flags non-chemical content.
- **Permissive mode**: Best-effort parsing — extracts what it can, warns about the rest. Used for single-formula prediction input.

---

## Feature Engineering

**File:** `packages/ml-core/piezo_ml/features/feature_engineer.py`

After parsing, each formula is transformed into a numerical feature vector. This is where domain science meets ML.

### Feature Vector Composition

```
Total features = Mole fractions + Weighted properties + Structural factors
               = N_elements    + (22 props × 2 stats) + 2
               = N_elements    + 44                     + 2
```

For a formula with 5 non-oxygen elements: `5 + 44 + 2 = 51 features`

### 1. Mole Fractions (`frac_<element>`)

For each non-oxygen element, the mole fraction relative to total non-oxygen stoichiometry:

```
frac_Ba = Ba_amount / (Ba + Ca + Ti + Zr + ... total non-O)
```

Oxygen is excluded from fractions because it's nearly constant in perovskite stoichiometries (always ~3). Including it would add noise rather than signal.

### 2. Weighted Physics Descriptors (44 features)

For each of the 22 numeric element properties (from the registry), two statistics are computed using mole-fraction-weighted averaging:

- **Weighted mean:** `Σ(property_i × fraction_i) / Σ(fraction_i)`
- **Weighted variance:** `Σ(fraction_i × (property_i - mean)²) / Σ(fraction_i)`

This produces 44 features like:
- `electronegativity_weighted_mean` — average electronegativity across elements
- `electronegativity_weighted_var` — spread of electronegativity (high variance → more ionic character mismatch)
- `ionic_radius_pm_weighted_mean` — average ionic radius
- `ionic_radius_pm_weighted_var` — radius mismatch (drives lattice strain)

### 3. Structural Factors (2 features)

Computed from A-site and B-site ionic radii (classified via `perovskite_site` in the registry):

- **Goldschmidt Tolerance Factor:** `t = (r_A + r_O) / (√2 × (r_B + r_O))`
  - `t ≈ 1.0`: ideal cubic perovskite
  - `t < 1.0`: tilted octahedra (rhombohedral/orthorhombic)
  - `t > 1.0`: hexagonal phases

- **Octahedral Factor:** `μ = r_B / r_O`
  - Indicates B-site ion stability within the oxygen octahedron

### 4. Composite Features (Optional — 8 features)

When `ENABLE_COMPOSITE_MODULE=true`, PVDF/polymer composite materials get additional encoded features:

| Feature | Type | Description |
|---------|------|-------------|
| `ceramic_type` | Categorical | Hard PZT, Soft PZT, KNN, BaTiO₃, etc. |
| `fabrication_method` | Categorical | Sol-gel, Solid-state, Hot pressing, etc. |
| `sintering_method` | Categorical | Conventional, SPS, Microwave, etc. |
| `sintering_temp_c` | Numeric | Sintering temperature (°C) |
| `matrix_type` | Categorical | PVDF, Epoxy, Silicone, etc. |
| `filler_wt_pct` | Numeric | Ceramic filler weight percentage |
| `particle_morphology` | Categorical | Spherical, Fibrous, Platelet, etc. |
| `particle_size_nm` | Numeric | Average particle size (nanometers) |
| `surface_treatment` | Categorical | Silane, None, Titanate, etc. |
| `relative_density_pct` | Numeric | Relative density (%) |

Categorical fields are encoded via the `composite_encoder.py` module. Valid options are enforced by `field_options_registry.py`.

---

## Supported Algorithms

All 8 algorithms are defined in `packages/ml-core/piezo_ml/models/algorithm_registry.py` (629 lines, 25KB).

### Algorithm Overview

| # | Algorithm | Key | Category | Notes |
|---|-----------|-----|----------|-------|
| 1 | **XGBoost** | `xgboost` | Gradient boosting | Gold standard for tabular data. L1+L2 regularization. |
| 2 | **Random Forest** | `random_forest` | Ensemble (bagging) | Robust to overfitting. No feature scaling needed. |
| 3 | **LightGBM** | `lightgbm` | Gradient boosting | Leaf-wise growth → faster training, lower memory. |
| 4 | **Gradient Boosting** | `gradient_boosting` | Gradient boosting | scikit-learn's native implementation. Staged predictions. |
| 5 | **SVR** | `svr` | Kernel methods | Wrapped in a StandardScaler Pipeline. Good for small datasets. |
| 6 | **Decision Tree** | `decision_tree` | Single tree | Fully interpretable. Fast. Prone to overfitting. |
| 7 | **Neural Network (ANN)** | `ann` | Deep learning | MLPRegressor. Wrapped in StandardScaler Pipeline. Captures non-linear patterns. |
| 8 | **Stacking Ensemble** | `stacking` | Meta-learning | Base: RF + XGBoost + SVR → Meta: Ridge regression. Most accurate, slowest. |

### Algorithm Selection Modes

The frontend offers three modes (visible in the Model Studio):

1. **Unified:** One algorithm for all targets (e.g., XGBoost for d₃₃, Tc, and Hardness)
2. **Per-Target:** Different algorithm per target (e.g., XGBoost for d₃₃, Random Forest for Tc, Stacking for Hardness)
3. **Auto-Tune:** Optuna finds the best algorithm + hyperparameters for each target (see [Auto-Tune](#auto-tune-optuna))

---

## Hyperparameter Reference

Every algorithm exposes tunable hyperparameters via sliders in the UI. Here's the complete reference:

### XGBoost

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_estimators` | 100 | 10–1000 | Number of boosting rounds |
| `max_depth` | 6 | 1–15 | Maximum tree depth |
| `learning_rate` | 0.1 | 0.001–1.0 | Step size shrinkage (eta) |
| `subsample` | 0.8 | 0.1–1.0 | Row sampling ratio per tree |
| `colsample_bytree` | 0.8 | 0.1–1.0 | Feature sampling ratio per tree |
| `min_child_weight` | 1 | 1–20 | Minimum sum of instance weight in a child |
| `gamma` | 0.0 | 0.0–5.0 | Minimum loss reduction for split |
| `reg_alpha` | 0.0 | 0.0–10.0 | L1 regularization (Lasso) |
| `reg_lambda` | 1.0 | 0.0–10.0 | L2 regularization (Ridge) |

### Random Forest

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_estimators` | 100 | 10–1000 | Number of trees |
| `max_depth` | None | 1–50 / None | Maximum tree depth (None = unlimited) |
| `min_samples_split` | 2 | 2–20 | Minimum samples to split a node |
| `min_samples_leaf` | 1 | 1–20 | Minimum samples in a leaf |
| `max_features` | `sqrt` | `sqrt`/`log2`/`auto`/float | Features considered per split |

### LightGBM

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_estimators` | 100 | 10–1000 | Number of boosting iterations |
| `max_depth` | -1 | -1–15 | Max depth (-1 = no limit) |
| `learning_rate` | 0.1 | 0.001–1.0 | Shrinkage factor |
| `num_leaves` | 31 | 10–200 | Maximum leaves per tree |
| `subsample` | 0.8 | 0.1–1.0 | Row sampling (bagging_fraction) |
| `colsample_bytree` | 0.8 | 0.1–1.0 | Feature sampling |
| `min_child_samples` | 20 | 5–100 | Min data in a leaf |
| `reg_alpha` | 0.0 | 0.0–10.0 | L1 regularization |
| `reg_lambda` | 0.0 | 0.0–10.0 | L2 regularization |

### Gradient Boosting (scikit-learn)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_estimators` | 100 | 10–1000 | Number of boosting stages |
| `max_depth` | 3 | 1–15 | Maximum tree depth |
| `learning_rate` | 0.1 | 0.001–1.0 | Step size |
| `subsample` | 1.0 | 0.1–1.0 | Row sampling |
| `min_samples_split` | 2 | 2–20 | Min samples for split |
| `min_samples_leaf` | 1 | 1–20 | Min samples in leaf |

### SVR (Support Vector Regression)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `kernel` | `rbf` | `linear`/`rbf`/`poly`/`sigmoid` | Kernel function |
| `C` | 1.0 | 0.001–1000 | Regularization (inverse) |
| `epsilon` | 0.1 | 0.001–1.0 | ε-insensitive tube width |
| `gamma` | `scale` | `scale`/`auto`/float | Kernel coefficient |

> SVR is wrapped in a `Pipeline([StandardScaler, SVR])` — feature scaling is automatic.

### Decision Tree

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_depth` | None | 1–50 / None | Maximum depth |
| `min_samples_split` | 2 | 2–20 | Min samples for split |
| `min_samples_leaf` | 1 | 1–20 | Min samples in leaf |
| `max_features` | None | `sqrt`/`log2`/None | Features per split |

### Neural Network (ANN)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hidden_layer_sizes` | `(100,)` | Varies | Tuple of layer widths (e.g., `(128, 64)`) |
| `activation` | `relu` | `relu`/`tanh`/`logistic` | Activation function |
| `solver` | `adam` | `adam`/`lbfgs`/`sgd` | Optimization algorithm |
| `alpha` | 0.0001 | 0.00001–1.0 | L2 penalty |
| `learning_rate_init` | 0.001 | 0.00001–0.1 | Initial learning rate |
| `max_iter` | 200 | 50–2000 | Maximum epochs |

> ANN is wrapped in a `Pipeline([StandardScaler, MLPRegressor])` — scaling is automatic.

### Stacking Ensemble

No user-tunable hyperparameters — the composition is fixed:
- **Base estimators:** Random Forest + XGBoost + SVR
- **Meta-learner:** Ridge Regression
- **Strategy:** 5-fold cross-validation for base predictions → meta-learner fits on stacked outputs

This consistently delivers the highest R² scores but is 3–5× slower than individual algorithms.

---

## Training Orchestration

**File:** `packages/ml-core/piezo_ml/pipeline/training_orchestrator.py` (23KB — the largest single file)

The orchestrator is the "brain" of the training pipeline. Here's what happens when you click "Train":

### Training Flow

```
1. Load Data
   └── Query materials from DB for selected dataset
   └── Filter by selected targets (d33, tc, vickers_hardness)

2. Apply Missing Value Strategies
   └── Per-column: KNN Imputer / Mean / Median / Mode / Drop rows
   └── Categorical fields: auto-fallback to Mode

3. Feature Engineering
   └── FormulaParser → FeatureEngineer → feature vectors
   └── Drop rows with parse failures
   └── Log skipped rows + reasons

4. Train/Test Split
   └── 80/20 split (stratified if possible)
   └── Save exact training data CSV as artifact

5. Per-Target Training Loop
   └── For each target (d33, tc, hardness):
       └── Build model (algorithm_registry.build_model)
       └── Fit on training data
       └── Evaluate on test data → R², RMSE
       └── Validate metrics (reject NaN/Inf)
       └── Save .joblib + metadata JSON
       └── Stream progress via WebSocket

6. Convergence Tracking
   └── For tree-based models: staged_predict per boosting round
   └── For ANN: loss_curve_ from MLPRegressor
   └── Sent to frontend for live chart updates

7. Results
   └── All models registered in DB (trained_models table)
   └── First model per target auto-set as default
   └── Training job status → "completed"
```

### Metric Validation

The trainer **rejects** any model producing non-finite metrics:

```python
# From trainer.py — strict validation
if not np.isfinite(r2) or not np.isfinite(rmse):
    raise ValueError(f"Model produced invalid metrics: R²={r2}, RMSE={rmse}")
```

This prevents corrupt models from entering the registry (a hard lesson from v2.1.0).

### Artifact Directory Structure

Each training run creates a timestamped artifact directory:

```
resources/training-artifacts/
└── 20260613_152801/
    ├── d33_train_data.csv              # Exact training data (reproducible)
    ├── d33_test_data.csv               # Exact test data
    ├── tc_train_data.csv
    ├── tc_test_data.csv
    ├── vickers_hardness_train_data.csv
    ├── vickers_hardness_test_data.csv
    └── training_config.json            # Full configuration snapshot
```

---

## Data Preprocessing & Missing Values

**File:** `packages/ml-core/piezo_ml/pipeline/missing_value_strategies.py`

The frontend exposes per-column missing value strategy selection (visible in the Model Studio "Missing Value Handling" section).

### Available Strategies

| Strategy | Numeric Columns | Categorical Columns | When to Use |
|----------|----------------|-------------------|-------------|
| **Drop** | Drop entire row | Drop entire row | Target columns with few missing values |
| **KNN Imputer** | k=3 nearest neighbors | Falls back to Mode | Best for numeric features with spatial relationships |
| **Mean** | Column mean | Falls back to Mode | Quick, when distribution is roughly normal |
| **Median** | Column median | Falls back to Mode | Robust to outliers |
| **Mode** | Most frequent value | Most frequent value | Categorical fields (always this) |

### Categorical Field Fallback

KNN, Mean, and Median are **mathematically meaningless** for categorical data (e.g., `sintering_method = "SPS"`). The system automatically falls back to **Mode** for any categorical field — this is enforced in `missing_value_strategies.py`, not in the frontend.

Categorical fields (from `field_registry.py`):
`ceramic_type`, `fabrication_method`, `sintering_method`, `matrix_type`, `particle_morphology`, `surface_treatment`

---

## Auto-Tune (Optuna)

**File:** `packages/ml-core/piezo_ml/models/optuna_tuner.py`

When "Auto-Tune" mode is selected, Optuna performs Bayesian hyperparameter optimization:

1. **Search space:** All hyperparameters from the reference above, per algorithm
2. **Objective:** Minimize negative R² (maximize R²) on cross-validation
3. **Trials:** 50 by default (configurable)
4. **Sampler:** TPE (Tree-structured Parzen Estimator)
5. **Pruning:** MedianPruner — kills underperforming trials early
6. **Output:** Best algorithm + hyperparameters per target

The tuner tests all 8 algorithms and returns the single best configuration per target. This means "Auto-Tune for d₃₃" might select XGBoost with `n_estimators=342, max_depth=8`, while "Auto-Tune for Tc" might select LightGBM with completely different hyperparameters.

---

## Model Registry & Versioning

Every trained model is persisted in two forms:

### 1. File System (`.joblib`)

```
resources/trained-models/
├── model_d33_xgboost_20260613_152801.joblib
├── model_d33_xgboost_20260613_152801_meta.json   # metadata
├── model_tc_random_forest_20260518_102225.joblib
├── model_tc_random_forest_20260518_102225_meta.json
└── ...
```

Metadata JSON contains:
```json
{
  "target": "d33",
  "algorithm": "xgboost",
  "r2_score": 0.827,
  "rmse": 45.17,
  "feature_version": "v2",
  "feature_dim": 91,
  "supported_elements": ["Ba", "Bi", "Ca", "K", ...],
  "training_samples": 211,
  "hyperparameters": { "n_estimators": 100, "max_depth": 6, ... },
  "created_at": "2026-06-13T15:28:01"
}
```

### 2. Database (`trained_models` table)

The DB record mirrors the metadata JSON plus:
- `is_default`: Boolean flag — the model used for predictions when no specific model is selected
- `dataset_id`: FK to the training dataset
- `training_job_id`: FK to the training job that produced it

### Default Model Selection

- The **first model trained** per target is automatically set as default
- Users can change the default via the Dashboard (star icon) or Settings
- The prediction engine always loads the `is_default=True` model for each target

---

**← [Setup Guide](SETUP_GUIDE.md)** | **Next: [Features Guide →](FEATURES_GUIDE.md)**
