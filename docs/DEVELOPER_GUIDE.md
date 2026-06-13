# Piezo.AI v2.2.0 — Developer Guide

> Complete technical reference for contributors, power users, and anyone who wants to understand every component of the platform.

This guide is split into focused sections. Start with what's relevant to you:

| Document | What's Inside |
|----------|--------------|
| **[Setup Guide](SETUP_GUIDE.md)** | Manual setup (with/without Docker), every environment variable, database configuration, version management |
| **[ML Pipeline](ML_PIPELINE.md)** | Central Element Registry, formula parsing, feature engineering, all 8 algorithms, training orchestration, hyperparameter reference |
| **[Features Guide](FEATURES_GUIDE.md)** | Prediction engine, SHAP interpretability, NSGA-II optimization, use-case mapping, symbolic regression, PDF reports, Settings |
| **[Interface Gallery](INTERFACE_GALLERY.md)** | All 22 interface screenshots with detailed descriptions of every section |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Logging architecture, diagnostics, common errors, macOS-specific fixes, known limitations |

---

## Architecture Overview

Piezo.AI is a **monorepo** managed by Turborepo with strict architectural boundaries. The golden rule:

> **FastAPI is a dumb pipe.** Zero ML logic is allowed in the API layer. All machine learning — parsing, feature engineering, training, inference, SHAP analysis, optimization — lives exclusively in `packages/ml-core/piezo_ml/`.

This separation means:
- The API layer is thin and testable — it handles HTTP, authentication, database access, and WebSocket streaming
- The ML core is a standalone Python package that can be used independently (e.g., from a Jupyter notebook)
- Schema changes flow through Alembic migrations in `packages/db/`

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser (localhost:3000)                      │
│  Next.js 15 + React 19 + TailwindCSS 4 + Zustand + Framer Motion  │
│                                                                     │
│  Pages: Dashboard │ Dataset │ Train │ Predict │ Optimization Lab   │
│         Interpretability │ Settings                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST API + WebSocket (SSE for training logs)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (localhost:8000)                 │
│                        "DUMB PIPE" — no ML logic                    │
│                                                                     │
│  Routers: /api/v1/datasets │ /training │ /predictions │ /dashboard │
│           /interpret │ /optimization │ /settings                    │
│                                                                     │
│  Core: Pydantic Settings │ Async SQLAlchemy │ Structured Logging   │
└───────────┬───────────────────────────────────┬─────────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌──────────────────────────────────┐
│  PostgreSQL 16        │         │  ML Core (piezo_ml)              │
│  (Docker or local)    │         │  Pure Python — no web framework  │
│                       │         │                                  │
│  Tables:              │         │  ├── registry/     (42 elements) │
│  - datasets           │         │  ├── parsers/      (formula)     │
│  - materials          │         │  ├── features/     (engineering) │
│  - training_jobs      │         │  ├── pipeline/     (orchestrate) │
│  - trained_models     │         │  ├── models/       (8 algorithms)│
│  - predictions        │         │  ├── evaluation/   (SHAP)        │
│  - prediction_batches │         │  ├── optimization/ (NSGA-II)     │
│  └──────────          │         │  ├── symbolic_regression/ (PySR) │
│                       │         │  ├── reporting/    (PDF)         │
│                       │         │  └── validators/   (post-parse)  │
└───────────────────────┘         └──────────────────────────────────┘
                                              │
                                              ▼
                                  ┌──────────────────────┐
                                  │  Filesystem Artifacts │
                                  │                      │
                                  │  resources/           │
                                  │  ├── trained-models/  │
                                  │  ├── training-artifacts│
                                  │  ├── shap-cache/      │
                                  │  ├── optimization-cache│
                                  │  └── prediction-results│
                                  └──────────────────────┘
```

### Data Flow: From Formula to Prediction

```
User types: "0.96(K₀.₄₈Na₀.₅₂)(Nb₀.₉₅Sb₀.₀₅)O₃–0.04Bi₀.₅Na₀.₅ZrO₃"
                           │
                           ▼
              ┌─── Pre-normalization ───┐
              │ Strip whitespace        │
              │ Unicode subscripts → ASCII │
              │ Validate strict mode    │
              └────────┬────────────────┘
                       ▼
              ┌─── Multi-phase Splitting ───┐
              │ DFS parenthesis scanner     │
              │ 0.96 × Phase1, 0.04 × Phase2 │
              └────────┬────────────────────┘
                       ▼
              ┌─── Element Extraction ───┐
              │ chemparse per phase      │
              │ Multiplier × counts      │
              │ Sum across phases        │
              └────────┬─────────────────┘
                       ▼
              ┌─── Feature Engineering ──────────────┐
              │ 42 element mole fractions             │
              │ 44 weighted physics descriptors       │
              │   (mean + variance for 22 properties) │
              │ Tolerance factor (Goldschmidt)        │
              │ Octahedral factor                     │
              │ + 8 composite features (if applicable)│
              │ → Final vector: 45–94 features        │
              └────────┬─────────────────────────────┘
                       ▼
              ┌─── Model Inference ───┐
              │ Load .joblib model    │
              │ model.predict(vector) │
              │ → d₃₃, Tc, Hardness  │
              └───────────────────────┘
```

### Database Schema (6 Tables)

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `datasets` | Uploaded CSV metadata | `display_name`, `status`, `total_rows`, `column_mapping` (JSONB), `has_composite_fields` |
| `materials` | Individual rows post-column-mapping | `formula`, `d33`, `tc`, `vickers_hardness`, `qm`, `kp`, composite fields, `parse_status`, `source_row`/`parsed_row` (JSONB) |
| `training_jobs` | Training pipeline execution state | `mode` (manual/auto), `targets` (JSONB), `algorithms` (JSONB), `progress_pct`, `current_stage`, `artifact_dir` |
| `trained_models` | Trained model metadata | `target`, `algorithm`, `r2_score`, `rmse`, `feature_version`, `feature_dim`, `supported_elements` (JSONB), `model_file_path`, `is_default` |
| `predictions` | Individual prediction history | `formula`, `d33_predicted`, `tc_predicted`, `hardness_predicted`, `d33_ci_lower`/`upper`, `prediction_status` |
| `prediction_batches` | Batch prediction job metadata | `source_filename`, `total_rows`, `success_count`, `error_count`, `result_file_path` |

All primary keys are UUIDs. Relationships enforce cascading deletes (deleting a dataset removes all its materials, jobs, and models).

### Monorepo Package Topology

```
piezo-ai (root)
├── @piezo-ai/web    (apps/web)    → Next.js frontend
├── @piezo-ai/api    (apps/api)    → FastAPI backend (not a Node package — managed by pip)
├── piezo-ml         (packages/ml-core) → Python ML package (pip install -e)
└── piezo-db         (packages/db)      → Python DB package (pip install -e)
```

- **Node workspace** (pnpm): manages `apps/web` only
- **Python packages**: installed via `pip install -e` in editable mode into a shared `.venv/`
- **Turborepo**: orchestrates `dev`, `build`, `lint`, `clean` tasks across workspaces

### Frontend State Management

The frontend uses **Zustand** stores for each section — no prop drilling, no context hell:

| Store | File | Manages |
|-------|------|---------|
| `dashboardStore` | `lib/store/dashboardStore.ts` | Dashboard stats, dataset list, model counts |
| `datasetStore` | `lib/store/datasetStore.ts` | Upload wizard state, column mapping, review issues |
| `trainingStore` | `lib/store/trainingStore.ts` | Algorithm selection, hyperparams, training progress |
| `predictStore` | `lib/store/predictStore.ts` | Formula input, prediction results, batch upload |
| `interpretStore` | `lib/store/interpretStore.ts` | SHAP mode selection, analysis results, physics validation |
| `optimizationStore` | `lib/store/optimizationStore.ts` | NSGA-II config, Pareto results, convergence data |
| `settingsStore` | `lib/store/settingsStore.ts` | Element registry, field schema, ML limits, theme |
| `uiStore` | `lib/store/uiStore.ts` | Sidebar collapse, mobile state |

API communication uses **TanStack Query** for server state with automatic caching and background refetching.

### Key Frontend Libraries

| Library | Purpose |
|---------|---------|
| `react-grid-layout` | Draggable/resizable dashboard cards (under development) |
| `recharts` | All charts (scatter, line, bar, pie, radar) |
| `framer-motion` | Page transitions, card animations, micro-interactions |
| `lucide-react` | Icon system |
| `@radix-ui/*` | Accessible primitives (dialog, dropdown, select, slider, switch, tabs, tooltip) |
| `@tanstack/react-table` | Virtualized data tables with sorting, filtering |
| `@tanstack/react-virtual` | Virtual scrolling for large datasets |
| `katex` | Mathematical formula rendering (SHAP equations, symbolic regression output) |
| `next-themes` | Theme switching (Dark, Light, Night) |
| `clsx` | Conditional class composition |

---

## Complete Project File Tree

```
Piezoelectrics-AI-Discovery-Lab/
├── apps/
│   ├── api/                          # FastAPI backend
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py              # App entrypoint, lifespan, CORS, router mounting
│   │   │   ├── core/
│   │   │   │   ├── config.py        # Pydantic Settings (reads .env)
│   │   │   │   ├── database.py      # Async SQLAlchemy engine + session factory
│   │   │   │   └── logging_config.py # Dual-output structured logging
│   │   │   └── modules/
│   │   │       ├── dashboard/       # GET stats, model list, dataset summary
│   │   │       ├── dataset/         # Upload, map columns, review issues, CRUD
│   │   │       ├── training/        # Start job, stream logs (SSE), get results
│   │   │       ├── prediction/      # Single + batch predict, history
│   │   │       ├── interpret/       # SHAP beeswarm/waterfall/dependence, physics, PySR
│   │   │       ├── optimization/    # NSGA-II run, presets, Pareto results
│   │   │       └── settings/        # Element registry CRUD, field schema, ML limits
│   │   ├── pyproject.toml           # Python deps: fastapi, uvicorn, sqlalchemy, etc.
│   │   └── package.json             # Placeholder for Turborepo discovery
│   └── web/                          # Next.js 15 frontend
│       ├── app/
│       │   ├── layout.tsx           # Root layout (ThemeProvider, sidebar, fonts)
│       │   ├── page.tsx             # Redirects to /dashboard
│       │   ├── globals.css          # Design system tokens + TailwindCSS 4
│       │   ├── dashboard/           # Dashboard page
│       │   ├── dataset/             # Dataset upload wizard + table explorer
│       │   ├── train/               # Model Studio (algorithm cards, hyperparams, logs)
│       │   ├── predict/             # Prediction page (single + batch)
│       │   ├── optimization-lab/    # Optimization Lab (Pareto + structure analysis)
│       │   ├── interpret/           # Interpretability (SHAP, physics, PySR)
│       │   └── settings/            # Settings (elements, fields, ML limits, theme)
│       ├── components/
│       │   ├── ui/                  # Shared: ChartNavigator, FormulaValidationInput, InfoTooltip
│       │   ├── common/              # Layout primitives
│       │   ├── layout/              # Sidebar, header, mobile navigation
│       │   ├── dashboard/           # Dashboard-specific components
│       │   ├── dataset/             # Upload wizard steps, data table
│       │   ├── train/               # Algorithm cards, hyperparameter sliders, log viewer
│       │   ├── predict/             # Formula input, results card, batch upload
│       │   ├── interpret/           # SHAP charts, physics validator, PySR viewer
│       │   ├── optimization/        # Pareto chart, convergence, preset cards
│       │   └── settings/            # Element registry table, field schema editor
│       ├── lib/
│       │   ├── constants.ts         # APP_CONFIG (branding, version, API URLs)
│       │   ├── api/                 # API client functions per module
│       │   ├── store/               # Zustand stores (8 stores)
│       │   ├── hooks/               # Custom React hooks
│       │   └── utils/               # Formatters, validators
│       ├── package.json             # Next.js, React 19, Radix, Recharts, etc.
│       ├── next.config.ts           # Next.js configuration
│       └── tsconfig.json            # TypeScript configuration
├── packages/
│   ├── ml-core/                      # ALL ML logic
│   │   ├── piezo_ml/
│   │   │   ├── __init__.py          # Package version + subpackage docstring
│   │   │   ├── registry/
│   │   │   │   ├── element_registry.py          # 42 elements, auto-bootstrap from mendeleev/pymatgen
│   │   │   │   ├── element_registry_data.json   # Pre-computed element properties (27 per element)
│   │   │   │   ├── element_classification.py    # A-site/B-site/dopant classification
│   │   │   │   ├── bootstrap_element.py         # New element bootstrapper
│   │   │   │   ├── field_schema_manager.py      # Dynamic field schema (customizable via Settings)
│   │   │   │   └── field_options_registry.py    # Valid options for categorical fields
│   │   │   ├── parsers/
│   │   │   │   ├── formula_parser.py            # Multi-phase solid solution parser
│   │   │   │   └── formula_normalizer.py        # Unicode normalization, validation
│   │   │   ├── features/
│   │   │   │   ├── feature_engineer.py          # Weighted physics descriptors, structural factors
│   │   │   │   ├── composite_encoder.py         # PVDF/polymer composite feature encoding
│   │   │   │   └── field_registry.py            # Supported field definitions
│   │   │   ├── pipeline/
│   │   │   │   ├── training_orchestrator.py     # Main training loop (23KB — the brain)
│   │   │   │   ├── preprocessing_pipeline.py    # Data cleaning + validation pipeline
│   │   │   │   ├── data_loader.py               # CSV → DataFrame with type coercion
│   │   │   │   ├── data_cleaner.py              # NaN/duplicate removal
│   │   │   │   ├── missing_value_strategies.py  # KNN/Mean/Median/Drop imputation
│   │   │   │   ├── sentinel_handler.py          # Composite sentinel value handling
│   │   │   │   ├── parsed_dataset_saver.py      # Save per-target training data CSVs
│   │   │   │   └── post_parse_validator.py      # Post-parse feature validation
│   │   │   ├── models/
│   │   │   │   ├── algorithm_registry.py        # 8 algorithms + hyperparameter definitions (25KB)
│   │   │   │   ├── trainer.py                   # fit/evaluate with metric validation
│   │   │   │   ├── inference_engine.py          # Load model + predict with CI
│   │   │   │   ├── model_saver.py               # .joblib serialization + metadata JSON
│   │   │   │   ├── optuna_tuner.py              # Bayesian hyperparameter optimization
│   │   │   │   ├── use_case_mapper.py           # 11-category Gaussian scoring engine
│   │   │   │   └── platform_utils.py            # Safe n_jobs for macOS OpenMP
│   │   │   ├── evaluation/
│   │   │   │   ├── shap_analyzer.py             # Beeswarm, Waterfall, Dependence
│   │   │   │   └── physics_validator.py         # SHAP vs solid-state physics checks
│   │   │   ├── optimization/
│   │   │   │   ├── nsga2_optimizer.py           # pymoo NSGA-II with ML surrogates
│   │   │   │   ├── pareto_utils.py              # Use-case tagging + Pareto ranking
│   │   │   │   └── structural_analyzer.py       # Crystal structure analysis
│   │   │   ├── symbolic_regression/             # PySR (Julia-based) integration
│   │   │   ├── reporting/                       # PDF report generation (ReportLab)
│   │   │   └── validators/                      # Formula validation utilities
│   │   ├── tests/                               # pytest test suite
│   │   └── pyproject.toml                       # scikit-learn, xgboost, shap, pymoo, etc.
│   └── db/
│       ├── piezo_db/
│       │   ├── __init__.py
│       │   ├── base.py              # SQLAlchemy declarative base
│       │   └── models.py            # 6 ORM models (datasets, materials, training_jobs, etc.)
│       ├── alembic/                 # Migration scripts
│       ├── alembic.ini              # Alembic configuration
│       └── pyproject.toml           # sqlalchemy, asyncpg, alembic, psycopg2-binary
├── resources/
│   ├── .env.defaults                # Fallback environment values
│   ├── main-datasets/               # Production CSV datasets
│   ├── sample-and-test-dataset/     # Sample CSVs for quick testing
│   ├── interface-previews/          # 22 UI screenshots
│   ├── trained-models/              # .joblib model artifacts (gitignored)
│   ├── training-artifacts/          # Per-run training data (gitignored)
│   ├── shap-cache/                  # Cached SHAP computations (gitignored)
│   ├── optimization-cache/          # Cached NSGA-II results (gitignored)
│   ├── prediction-results/          # Batch prediction output CSVs (gitignored)
│   └── reports/                     # Generated PDF reports (gitignored)
├── scripts/
│   ├── dev.sh                       # Main dev utility (507 lines)
│   └── lib/
│       ├── _colors.sh               # Terminal color definitions
│       ├── _python.sh               # Python/pyenv setup (14KB)
│       ├── _node.sh                 # Node.js/nvm setup (7KB)
│       ├── _database.sh             # PostgreSQL/Docker management (10KB)
│       └── _network.sh              # Network diagnostics (5KB)
├── docker/
│   └── docker-compose.yml           # PostgreSQL 16 Alpine container
├── .env.example                     # Environment variable template
├── .nvmrc                           # Node.js version (20)
├── .python-version                  # Python version (3.13)
├── package.json                     # Root workspace (Turborepo)
├── pnpm-workspace.yaml              # pnpm workspace definition
├── turbo.json                       # Turborepo task configuration
└── requirements.txt                 # Legacy requirements (use pyproject.toml instead)
```

---

**Next:** [Setup Guide →](SETUP_GUIDE.md) | [ML Pipeline →](ML_PIPELINE.md) | [Features Guide →](FEATURES_GUIDE.md) | [Interface Gallery →](INTERFACE_GALLERY.md) | [Troubleshooting →](TROUBLESHOOTING.md)
