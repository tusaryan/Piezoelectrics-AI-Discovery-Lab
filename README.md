# Piezo.AI v2.2.0

> **A local-first AI laboratory that replaces months of materials experimentation with millisecond predictions — helping researchers discover lead-free piezoelectric materials without writing a single line of code.**

<div align="center">
  <img src="resources/interface-previews/interface-preview-1.png" width="90%" alt="Piezo.AI Dashboard — showing live dataset counts, trained models, quick actions, and target distribution chart" />
  <br/><br/>
  <img src="resources/interface-previews/interface-preview-12.png" width="90%" alt="Optimization Lab — NSGA-II Pareto front visualization with use-case presets and convergence tracking" />
  <br/><br/>
  <em>Dashboard overview and Optimization Lab — See the <a href="docs/INTERFACE_GALLERY.md">Interface Gallery</a> for all 22 annotated screenshots.</em>
</div>

---

## Why Piezo.AI Exists

Lead Zirconate Titanate (PZT) is the backbone of nearly every modern piezoelectric device — from medical ultrasound probes to smartphone haptics. But PZT is over **60% lead by weight**, a neurotoxin that bioaccumulates and is increasingly banned under EU RoHS, REACH, and WEEE regulations.

The problem? Finding a replacement isn't trivial. Piezoelectric performance depends on complex interactions between dozens of elements, and each new composition takes **2–6 weeks** to synthesize and test in a lab. With millions of possible combinations across material families like KNN, BaTiO₃, BNT, BCZT, and BiFeO₃, exhaustive lab exploration is simply infeasible.

**Piezo.AI changes the equation.** Instead of synthesizing every candidate, you train ML models on existing data and screen thousands of compositions in seconds. The platform tells you which formulas are worth synthesizing — acting as a high-pass filter that eliminates 99% of poor candidates before you ever heat up a furnace.

### What This Means in Practice

| Metric | Traditional Lab | Piezo.AI | Speedup |
|--------|----------------|----------|---------|
| Compositions screened per session | 1–5 | ~5,000 | **1,000×** |
| Time per composition evaluation | 2–6 weeks | < 1 millisecond | **~10⁷×** |
| Cost per screening cycle | ₹5–25 lakh | ₹0 (runs locally) | **∞** |
| Trade-off analysis | Manual, qualitative | Automated Pareto front | Systematic |
| Data privacy | Variable (cloud tools) | Zero risk (fully local) | Eliminated |

---

## What You Can Actually Do With It

**Upload your data → Train models → Predict properties → Optimize compositions → Understand why.**

That's the workflow, from start to finish, all in the browser.

### 1. Upload & Manage Datasets
Drop a CSV with chemical formulas and measured properties. The system auto-detects columns, validates chemical formulas against 42 supported elements, flags data quality issues, and maps everything to its internal schema. Supports bulk ceramics, PVDF composites, and multi-phase solid solutions — no format gymnastics needed.

### 2. Train ML Models — No Code Required
Pick your target properties (d₃₃, Tc, Vickers Hardness), choose from **8 algorithms** (XGBoost, Random Forest, LightGBM, Gradient Boosting, SVR, Decision Tree, Neural Network, Stacking Ensemble), and tune hyperparameters via sliders. Or select Auto-Tune and let Optuna find the best configuration for you. Training logs stream live to the browser terminal, convergence charts update in real-time, and every trained model is versioned in a registry with one-click activation.

### 3. Predict Properties Instantly
Type a chemical formula — even a complex solid solution like `0.96(K₀.₄₈Na₀.₅₂)(Nb₀.₉₅Sb₀.₀₅)O₃–0.04Bi₀.₅Na₀.₅ZrO₃` — and get d₃₃, Tc, and Hardness predictions with 95% confidence intervals in milliseconds. Upload a CSV for batch predictions across hundreds of formulas at once.

### 4. Understand What Drives Predictions (Interpretability)
Don't trust black boxes? Three SHAP analysis modes (Beeswarm, Waterfall, Dependence) decompose every prediction into individual feature contributions. A Physics Validator automatically checks whether the model's logic aligns with established solid-state physics. PySR symbolic regression discovers compact mathematical equations (rendered in KaTeX) that approximate the model's learned relationships.

### 5. Optimize Across Competing Objectives (Optimization Lab)
Maximizing d₃₃ usually lowers Tc — that's the fundamental trade-off in piezoelectric design. The Optimization Lab uses NSGA-II to evolve ~5,000 compositions across 50 generations in under 10 seconds, generating a Pareto front of optimal trade-offs. Use-case presets (Flexible Wearables, Industrial Actuators, Ultrasonic Transducers) automatically configure target ranges.

### 6. Map to Real-World Applications
An 11-category scoring engine maps predicted properties to specific applications — Medical Ultrasound, Energy Harvesting, Wearable IoT, Aerospace SHM, Precision MEMS, and more — using Gaussian-weighted scoring with confidence tiers (Primary/Secondary/Tertiary).

---

## Prerequisites

| Tool | Required Version | Why |
|------|-----------------|-----|
| **Python** | **3.13.x** (3.11–3.13 accepted) | `mendeleev` requires `<3.14`. Python 3.14 will **not** work. |
| **Node.js** | **20.x** (LTS) | Next.js 15 requires Node 20+. |
| **pnpm** | 10+ | Monorepo workspace package manager. |
| **Docker Desktop** | Latest (or local PostgreSQL 16) | Database container. |

> **Version auto-detection:** The repo includes `.python-version` (pyenv) and `.nvmrc` (nvm) files. If you use these tools, running `pyenv local` and `nvm use` in the project root automatically selects the correct versions.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/tusaryan/Piezoelectrics-AI-Discovery-Lab.git
cd Piezoelectrics-AI-Discovery-Lab

# 2. Set correct Node version (if using nvm)
nvm use

# 3. Full setup — creates venv, installs Python/Node deps, starts DB, runs migrations
bash scripts/dev.sh setup

# 4. Start development servers
bash scripts/dev.sh start
```

This will:
1. Create a Python 3.13 virtual environment (`.venv/`) — nothing installs globally
2. Install all Python packages (FastAPI, scikit-learn, XGBoost, SHAP, pymoo, etc.)
3. Install frontend dependencies via pnpm
4. Start PostgreSQL via Docker (or connect to your local instance)
5. Run database migrations (Alembic)
6. Launch **FastAPI** on `http://localhost:8000`
7. Launch **Next.js** on `http://localhost:3000`

Open `http://localhost:3000` and you're in.

### Without Docker (Local PostgreSQL)

If you'd rather skip Docker entirely:

```bash
# 1. Create the database manually
createdb piezo_ai
# Or via psql:
# psql -c "CREATE DATABASE piezo_ai;"

# 2. Update DATABASE_URL in .env
#    DATABASE_URL=postgresql+asyncpg://your_user:your_pass@localhost:5432/piezo_ai

# 3. Run setup (choose "Local PostgreSQL" when prompted)
bash scripts/dev.sh setup

# 4. Start
bash scripts/dev.sh start
```

---

## Development Commands

```bash
bash scripts/dev.sh <command>
```

| Command | What it does |
|---------|-------------|
| `setup` | Incremental setup — keeps existing deps if compatible |
| `setup:all` | Full clean + fresh install (wipes `.venv`, `node_modules`) |
| `clean` | Remove `node_modules`, `.next`, `__pycache__`, `.venv` |
| `start` | Start backend + frontend dev servers |
| `stop` | Gracefully shut down all servers + free ports |
| `db:create` | Create the PostgreSQL database |
| `db:migrate` | Run Alembic schema migrations |
| `db:reset` | Drop and recreate DB + run migrations (**destroys all data**) |
| `diagnose` | Run full diagnostics (Python, Node, network, DB connectivity) |

---

## Project Structure

```
Piezoelectrics-AI-Discovery-Lab/
├── apps/
│   ├── api/                  # FastAPI backend (DUMB PIPE — zero ML logic)
│   │   └── app/
│   │       ├── core/         # Config, database, structured logging
│   │       └── modules/      # Dataset, Training, Prediction, Interpret,
│   │                         # Optimization, Dashboard, Settings routers
│   └── web/                  # Next.js 15 + React 19 frontend
│       ├── app/              # Pages (App Router): dashboard, dataset, train,
│       │                     # predict, optimization-lab, interpret, settings
│       ├── components/       # UI components per section + shared primitives
│       └── lib/              # API clients, Zustand stores, hooks, constants
├── packages/
│   ├── ml-core/              # ALL ML logic lives here — no exceptions
│   │   └── piezo_ml/
│   │       ├── registry/     # Central Element Registry (42 elements, 27 properties)
│   │       ├── parsers/      # Formula parsing + normalization
│   │       ├── features/     # Feature engineering + composite encoding
│   │       ├── pipeline/     # Data loading, cleaning, training orchestration
│   │       ├── models/       # Algorithm registry, trainer, inference, use-case mapper
│   │       ├── evaluation/   # SHAP analyzer + physics validator
│   │       ├── optimization/ # NSGA-II optimizer + Pareto utilities
│   │       ├── symbolic_regression/  # PySR integration
│   │       ├── reporting/    # PDF report generation (ReportLab)
│   │       └── validators/   # Post-parse validation
│   └── db/                   # SQLAlchemy models + Alembic migrations
├── resources/
│   ├── main-datasets/        # Source CSV datasets
│   ├── sample-and-test-dataset/ # Sample datasets for testing
│   ├── trained-models/       # Saved .joblib model artifacts
│   ├── training-artifacts/   # Per-run training data snapshots
│   ├── shap-cache/           # SHAP computation cache
│   └── optimization-cache/   # NSGA-II result cache
├── scripts/
│   ├── dev.sh                # Main dev utility script
│   └── lib/                  # Modular shell libraries (_python.sh, _node.sh, etc.)
├── docker/
│   └── docker-compose.yml    # PostgreSQL 16 container
├── docs/                     # Extended developer documentation
└── .env.example              # Environment variable template
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 15, React 19, TailwindCSS 4, Framer Motion, Zustand, Radix UI, Recharts, TanStack Table/Query, KaTeX |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy 2 (async), Pydantic v2, WebSockets |
| **Database** | PostgreSQL 16 (Docker or local) |
| **ML Core** | scikit-learn, XGBoost, LightGBM, SHAP, Optuna, pymoo (NSGA-II), PySR |
| **Chemistry** | chemparse, pymatgen, mendeleev |
| **Reporting** | ReportLab (PDF), Matplotlib |
| **Build** | Turborepo, pnpm workspaces, pyenv, nvm |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `mendeleev` / `pymatgen` install fails | You're on Python 3.14. Switch: `python3.13 -m venv .venv` |
| `pnpm: command not found` | `npm install -g pnpm` |
| Port 8000/3000 already in use | `bash scripts/dev.sh stop` or `lsof -ti:8000 \| xargs kill -9` |
| Docker permission denied | Make sure Docker Desktop is running |
| `alembic` path confusion | Always run from project root: `alembic -c packages/db/alembic.ini upgrade head` |
| SHAP/XGBoost crashes on macOS | The platform auto-sets `OMP_NUM_THREADS=1` — if it persists, run `PZ_LOG_LEVEL=DEBUG bash scripts/dev.sh start` for diagnostics |

---

## Detailed Documentation

For power users, contributors, and anyone who wants to understand every knob:

| Document | What's Inside |
|----------|--------------|
| **📖 [Developer Guide](docs/DEVELOPER_GUIDE.md)** | Architecture overview, system diagrams, database schema, monorepo topology, complete file tree |
| **🔧 [Setup Guide](docs/SETUP_GUIDE.md)** | Manual setup, every environment variable, Docker/local DB config, dev.sh commands, version management |
| **🧪 [ML Pipeline](docs/ML_PIPELINE.md)** | Element Registry (42 elements, 27 properties), formula parsing, feature engineering, all 8 algorithms with full hyperparameter tables, training orchestration |
| **⚡ [Features Guide](docs/FEATURES_GUIDE.md)** | Prediction engine, SHAP interpretability, NSGA-II optimization, use-case mapping (11 categories), symbolic regression, PDF reports, Settings |
| **🖼️ [Interface Gallery](docs/INTERFACE_GALLERY.md)** | All 22 interface screenshots with detailed descriptions |
| **🔍 [Troubleshooting](docs/TROUBLESHOOTING.md)** | Logging architecture, diagnostics, common errors + fixes, macOS-specific issues, known limitations |

---

## Acknowledgments

- **Dr. Kaustubh Kambale** — Project Supervisor, Department of Metallurgical & Materials Engineering, PEC Chandigarh
- **Dr. Sumeet Kumar Sharma** — Former Faculty, PEC Chandigarh
- Built as a B.Tech minor project at Punjab Engineering College (Deemed to be University), Chandigarh
- Dataset sources: The Materials Project, Crystallography Open Database (COD), and peer-reviewed literature

## License

This project is part of academic research at Punjab Engineering College, Chandigarh.
