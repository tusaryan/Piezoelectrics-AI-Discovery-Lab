# 🧪 Piezo.AI — AI-Driven Discovery of Lead-Free Piezoelectrics

![Project Status](https://img.shields.io/badge/Status-Active_Development-green)
![Tech Stack](https://img.shields.io/badge/Stack-Next.js_FastAPI_PostgreSQL-blue)
![Domain](https://img.shields.io/badge/Domain-Materials_Informatics-purple)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![Node](https://img.shields.io/badge/Node.js-18+-339933)

## 📖 Overview

**Piezo.AI** is a full-stack Materials Informatics platform that uses machine learning to accelerate the discovery of high-performance, lead-free piezoelectric materials. It replaces months of traditional trial-and-error experimentation with millisecond predictions.

### Synopsis Objectives (6th Semester)

| # | Objective | Status |
|---|-----------|--------|
| 1 | **Domain Expansion** — Augment KNN dataset with PVDF-KNN composite data | ✅ Backend ready |
| 2 | **Mechanical Property Prediction** — ML models for hardness (Vickers/Mohs) | ✅ Backend ready |
| 3 | **Model Transparency** — SHAP explainability | 🔜 Future work |
| 4 | **Structural Analysis AI** — Pre-trained models for crystal structures | 🔜 Future work |

---

## 🚀 Features

### Core Platform
- **Multi-Model Training** — Train and compare Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM, and more
- **Dual Training Mode** — Auto-Intelligent (benchmarks all models) or Expert Manual (fine-tune specific algorithms)
- **Real-Time Training Terminal** — Live log streaming via SSE to the browser terminal
- **Instant Prediction** — Predict d33 and Tc from chemical formulas in milliseconds
- **Batch Prediction** — Upload CSV files for bulk predictions
- **PDF Report Generation** — One-click export of comprehensive analysis reports

### Extended Capabilities
- **PVDF Composite Prediction** — Predict properties of ceramic-polymer composites with configurable filler %, morphology, and processing methods
- **Hardness Prediction & Use-Case Mapping** — Vickers/Mohs hardness estimation with automatic industrial application classification
- **Interactive Dataset Management** — Upload, view, and manage training datasets via the web UI
- **Model Registry** — Track all trained models with metrics, activate best performers

### Future Work
- **SHAP Interpretability** — Feature attribution analysis for model transparency
- **GNN Transfer Learning** — Crystal structure-aware predictions via pre-trained graph neural networks
- **Multi-Objective Optimization** — Pareto front mapping for d33 vs Tc vs Hardness
- **Active Learning** — Smart experiment suggestion to reduce lab iterations

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** — High-performance async Python web framework
- **PostgreSQL** — Relational database for datasets, training jobs, and model artifacts
- **scikit-learn, XGBoost, LightGBM** — ML model training and prediction
- **Alembic** — Database migrations
- **Joblib** — Model serialization

### Frontend
- **Next.js 14** — React framework with App Router
- **Shadcn/UI** — Modern component library
- **TypeScript** — Type-safe frontend code

### Infrastructure
- **Docker** — Optional containerized setup
- **PostgreSQL** — Required (local or Docker)

---

## 💻 Local Development Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Backend & ML |
| **Node.js** | 18+ | Frontend |
| **PostgreSQL** | 14+ | Database |
| **pnpm** | 8+ | **Strictly Required** for dependencies |

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/tusaryan/Piezoelectrics-AI-Discovery-Lab.git
cd Piezoelectrics-AI-Discovery-Lab

# 2. Run the setup script (use `setup:all` for a fresh start with cleanup)
bash scripts/dev.sh setup

# 3. Start development servers (must run from project root!)
bash scripts/dev.sh start
```

### Manual Setup (Step-by-Step)

> **⚠️ Important:** All commands below must be run from the **project root** directory (`Piezoelectrics-AI-Discovery-Lab/`) unless stated otherwise.

#### 1. Environment Variables
```bash
# 📂 Run from: project root
cp .env.example .env
# Edit .env with your PostgreSQL credentials if different from defaults
```

#### 2. Backend Setup (Python + venv)
```bash
# 📂 Run from: project root
python3 -m venv apps/api/.venv
source apps/api/.venv/bin/activate  # macOS/Linux
# apps\api\.venv\Scripts\activate   # Windows

# 📂 Run from: project root (with venv activated)
pip install -e apps/api
pip install -e packages/ml-core
pip install -e packages/db
```

#### 3. Node.js Setup (using nvm)
```bash
# 📂 Run from: project root
# Install nvm (if not installed)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Install and use Node 18+
nvm install 18
nvm use 18

# 🛑 Install pnpm (STRICTLY REQUIRED, npm/yarn are blocked)
npm install -g pnpm

# Install dependencies
pnpm install
```

#### 4. Database Setup
```bash
# 📂 Run from: project root

# Option A: Local PostgreSQL
createdb piezo_ai  # or use pgAdmin

# Option B: Docker PostgreSQL
docker compose -f docker/docker-compose.dev.yml up -d

# Run migrations
bash scripts/dev.sh db:migrate
```

#### 5. Start Development
```bash
# 📂 Terminal 1 — Backend (from project root)
source apps/api/.venv/bin/activate
uvicorn apps.api.app.main:app --reload --port 8000

# 📂 Terminal 2 — Frontend (from apps/web/)
cd apps/web
npm run dev
```

| Service | URL |
|---------|-----|
| Web App | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/v1/health |

---

## 🗂️ Project Structure

```
Piezoelectrics-AI-Discovery-Lab/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   └── app/
│   │       ├── core/           # Config, DB, error handling
│   │       └── modules/        # Feature modules
│   │           ├── training/   # ML training pipeline
│   │           ├── prediction/ # Property prediction
│   │           ├── dataset/    # Dataset management
│   │           ├── composite/  # PVDF composite predictions
│   │           ├── hardness/   # Hardness & use-case mapping
│   │           ├── interpret/  # SHAP interpretability
│   │           ├── inverse/    # Inverse design
│   │           └── active_learning/
│   └── web/                    # Next.js frontend
│       ├── app/                # Pages (App Router)
│       └── components/         # UI components
├── packages/
│   ├── db/                     # Database models & migrations
│   └── ml-core/                # ML pipeline (piezo_ml)
├── resources/                  # Sample datasets
│   ├── sample_knn_basic.csv
│   ├── sample_knn_pvdf_composite.csv
│   └── sample_hardness_only.csv
├── scripts/                    # Dev utility scripts
│   └── dev.sh                  # setup, clean, db:reset, start
└── .env.example                # Environment template
```

---

## 📊 Dataset Guide

### Uploading Datasets
Navigate to the **Dataset** page in the web UI and upload a `.csv` file. The system automatically detects available columns and maps them to prediction targets.

### Supported Schemas

**Basic (d33 + Tc only):**
```csv
formula,d33,tc
KNbO3,66.4,435
K0.5Na0.5NbO3,151.0,420
```

**Extended (with Hardness):**
```csv
formula,d33,tc,vickers_hardness
KNbO3,66.4,435,510.0
K0.5Na0.5NbO3,151.0,420,480.0
```

**Full Composite Schema:**
```csv
formula,d33,tc,vickers_hardness,matrix_type,filler_wt_pct,particle_morphology,particle_size_nm,surface_treatment,fabrication_method
K0.5Na0.5NbO3,58.0,118,32.0,pvdf,15.0,spherical,80,silane,solvent_cast
```

### Column Reference

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `formula` | string | ✅ | Chemical composition (supports solid solutions) |
| `d33` | float | For d33 training | Piezoelectric coefficient (pC/N) |
| `tc` | float | For Tc training | Curie temperature (°C) |
| `vickers_hardness` | float | For hardness training | Vickers hardness (HV) |
| `matrix_type` | string | For composites | Polymer matrix: `pvdf`, `pvdf_trfe`, `epoxy` |
| `filler_wt_pct` | float | For composites | Ceramic filler weight % (0-80) |
| `particle_morphology` | string | For composites | `spherical`, `rod`, `platelet` |
| `particle_size_nm` | float | For composites | Average particle size in nm |
| `surface_treatment` | string | For composites | `untreated`, `silane`, `plasma` |
| `fabrication_method` | string | For composites | `solvent_cast`, `electrospinning`, `hot_press` |

### Sample Datasets

Pre-made sample files are in `resources/` for quick testing:
- `sample_knn_basic.csv` — 20 KNN compositions with d33 and Tc
- `sample_knn_pvdf_composite.csv` — Bulk + composite data with all fields
- `sample_hardness_only.csv` — 15 compositions with Vickers hardness

---

## 🔧 Dev Utility Commands

```bash
bash scripts/dev.sh setup      # Incremental setup (skips cleanup)
bash scripts/dev.sh setup:all  # Full fresh setup (cleans caches/node_modules first)
bash scripts/dev.sh clean      # Remove node_modules, .next, __pycache__, .venv
bash scripts/dev.sh db:create  # Create the database
bash scripts/dev.sh db:reset   # Drop and recreate DB + run migrations
bash scripts/dev.sh db:migrate # Run Alembic migrations only
bash scripts/dev.sh db:seed    # Show info about sample datasets
bash scripts/dev.sh start      # Start backend + frontend dev servers
```

---

## 📜 License

This project is licensed under the MIT License.

## 🤝 Acknowledgments

- **Dr. Sumeet Kumar Sharma** — Project Mentor, PEC Chandigarh
- Based on KNN-based ceramics research methodologies
- Inspired by recent advancements in ML-assisted materials discovery
