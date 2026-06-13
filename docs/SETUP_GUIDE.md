# Setup Guide

> Complete manual setup instructions, environment variable reference, database configuration, and individual component management.

**← [Back to Developer Guide](DEVELOPER_GUIDE.md)**

---

## Table of Contents

- [Automated Setup (Recommended)](#automated-setup-recommended)
- [Manual Setup (Component by Component)](#manual-setup-component-by-component)
- [Environment Variable Reference](#environment-variable-reference)
- [Database Configuration](#database-configuration)
- [dev.sh Command Reference](#devsh-command-reference)
- [Running Individual Components](#running-individual-components)
- [Version Management](#version-management)

---

## Automated Setup (Recommended)

The `scripts/dev.sh` script handles everything — Python version, Node version, virtual environment, dependencies, database, and migrations. This is the recommended path.

```bash
# Clone and enter
git clone https://github.com/tusaryan/Piezoelectrics-AI-Discovery-Lab.git
cd Piezoelectrics-AI-Discovery-Lab

# Copy environment template
cp .env.example .env
# Edit .env if you want to change database credentials, API keys, etc.

# Full setup
bash scripts/dev.sh setup

# Start all services
bash scripts/dev.sh start
```

**What `setup` does, step by step:**

1. **Detects your OS** (macOS, Linux)
2. **Checks/installs pyenv** — if missing, offers to install it
3. **Installs Python 3.13.x** via pyenv (reads `.python-version`)
4. **Creates `.venv/`** using the correct Python binary — never pollutes your global environment
5. **Installs Python packages** from `packages/ml-core/pyproject.toml`, `packages/db/pyproject.toml`, and `apps/api/pyproject.toml` in editable mode (`pip install -e`)
6. **Checks/installs nvm** — if missing, offers to install it
7. **Installs Node.js 20** via nvm (reads `.nvmrc`)
8. **Installs pnpm** globally via npm
9. **Runs `pnpm install`** in the workspace root
10. **Starts PostgreSQL** (via Docker or detects local instance)
11. **Runs Alembic migrations** (`alembic upgrade head`)

**What `start` does:**

1. Verifies `.venv/` exists and is compatible
2. Starts PostgreSQL container (if using Docker)
3. Launches FastAPI backend via `uvicorn` (port 8000) in the background
4. Launches Next.js dev server via `pnpm dev` (port 3000) in the background
5. Streams both logs to your terminal with color-coded prefixes

---

## Manual Setup (Component by Component)

If you prefer to manage each piece yourself, or need to set up only part of the stack:

### 1. Python Environment

```bash
# Option A: Using pyenv (recommended)
pyenv install 3.13.3  # or any 3.13.x
pyenv local 3.13.3

# Option B: Using system Python (must be 3.11–3.13)
python3.13 --version  # verify

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install Python packages (editable mode)
pip install -e packages/db
pip install -e packages/ml-core
pip install -e apps/api

# Verify
python -c "import piezo_ml; print('ML Core OK')"
python -c "import piezo_db; print('DB OK')"
```

> **Why Python 3.13?** The `mendeleev` package (used for element property lookups) requires Python `< 3.14`. Python 3.14 **will not work** — the install will fail immediately.

### 2. Node.js Environment

```bash
# Option A: Using nvm (recommended)
nvm install 20
nvm use 20

# Option B: Using system Node (must be 20+)
node --version  # verify v20.x

# Install pnpm
npm install -g pnpm

# Install workspace dependencies
pnpm install
```

### 3. Database Setup

See [Database Configuration](#database-configuration) below.

### 4. Starting Individual Servers

```bash
# Backend only
source .venv/bin/activate
cd apps/api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend only (separate terminal)
cd apps/web
pnpm dev
```

---

## Environment Variable Reference

Create a `.env` file in the project root. Every variable has sensible defaults — you can run with zero changes for local development.

### Core Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_VERSION` | `2.2.0` | Displayed in the sidebar footer |
| `APP_NAME` | `Piezo.AI` | Application display name |
| `DEBUG` | `false` | Enables FastAPI debug mode, verbose SQL logging |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai` | Async SQLAlchemy connection string. Must use `asyncpg` driver. |

**Format:** `postgresql+asyncpg://<user>:<password>@<host>:<port>/<database>`

### API & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON array of allowed CORS origins |
| `API_SECRET_KEY` | `change-me-to-random-256-bit-secret` | API secret key — change in production |

### ML Artifact Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ARTIFACTS_PATH` | `./resources/trained-models` | Where `.joblib` model files are saved |
| `TRAINING_ARTIFACTS_PATH` | `./resources/training-artifacts` | Where per-run training data CSVs are stored |

### Feature Flags

Feature flags enable/disable entire modules at the backend level. When disabled, the corresponding API endpoints return 404 and the frontend hides related UI.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_COMPOSITE_MODULE` | `true` | Enables PVDF/polymer composite material support (ceramic_type, matrix_type, filler_wt_pct, particle_size_nm, particle_morphology, fabrication_method, sintering_method, sintering_temp_c, surface_treatment, relative_density_pct) |
| `ENABLE_HARDNESS_MODULE` | `true` | Enables Vickers Hardness (HV) as a trainable/predictable target |
| `ENABLE_GNN_MODULE` | `false` | **Deferred — v3 roadmap.** Graph Neural Network module for crystal structure-aware predictions |
| `ENABLE_AGENT_MODULE` | `false` | **Deferred — v3 roadmap.** Autonomous experimental planning agent |

### LLM Configuration (Optional)

These settings enable AI-generated insights in PDF reports and interpretation summaries. The platform is fully functional without any LLM — these are additive.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `""` (disabled) | Provider: `openai`, `anthropic`, `google`, `ollama` |
| `LLM_MODEL` | `""` | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.5-pro`, `llama3.1:8b`) |
| `LLM_API_KEY` | `""` | API key for the selected provider |
| `LLM_BASE_URL` | `""` | Custom base URL (required for Ollama: `http://localhost:11434`) |
| `LLM_TEMPERATURE` | `0.1` | Response randomness (0.0–1.0). Low = deterministic scientific output |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM response |
| `GEMINI_API_KEY` | `""` | Alias — auto-used if `LLM_API_KEY` is empty |

### Frontend Variables (NEXT_PUBLIC_*)

These are exposed to the browser. They configure branding, API endpoints, and feature toggles.

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI backend URL |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000` | WebSocket URL for real-time training logs |
| `NEXT_PUBLIC_APP_VERSION` | `2.1.0` | Version shown in sidebar footer |
| `NEXT_PUBLIC_APP_NAME` | `Piezo.AI` | App display name (sidebar, header, mobile) |
| `NEXT_PUBLIC_APP_LOGO_TEXT` | `P` | Fallback logo text when image fails |
| `NEXT_PUBLIC_APP_LOGO_PATH` | `/piezo-ai-logo.png` | Logo image path (relative to `public/`) |
| `NEXT_PUBLIC_ENABLE_COMPOSITE` | `true` | Show/hide composite fields in frontend |
| `NEXT_PUBLIC_ENABLE_HARDNESS` | `true` | Show/hide hardness target in frontend |
| `NEXT_PUBLIC_ENABLE_GNN` | `false` | Show/hide GNN module (deferred) |
| `NEXT_PUBLIC_ENABLE_AGENT` | `false` | Show/hide Agent module (deferred) |
| `NEXT_PUBLIC_DEV_NAME` | `Aryan` | Developer name (sidebar footer) |
| `NEXT_PUBLIC_DEV_GITHUB` | `https://github.com/tusaryan` | GitHub link (sidebar footer) |
| `NEXT_PUBLIC_DEV_LINKEDIN` | `https://linkedin.com/in/tusaryan` | LinkedIn link (sidebar footer) |

### Logging Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PZ_LOG_LEVEL` | `INFO` | Minimum log level for terminal: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ENABLE_VOICE` | `false` | **Deferred.** Voice interaction module |

---

## Database Configuration

### Option A: Docker (Default)

Docker Compose spins up a PostgreSQL 16 Alpine container with persistent data:

```bash
# Start database container
docker compose -f docker/docker-compose.yml up -d

# Verify it's running
docker ps  # should show 'piezo-ai-db'

# Check health
docker exec piezo-ai-db pg_isready -U piezo -d piezo_ai
```

**Docker Compose details:**

| Property | Value |
|----------|-------|
| Image | `postgres:16-alpine` |
| Container | `piezo-ai-db` |
| User | `piezo` |
| Password | `piezo` |
| Database | `piezo_ai` |
| Port | `5432` |
| Volume | `piezo-ai-pgdata` (persistent) |
| Healthcheck | `pg_isready` every 10s |

**To reset the database completely:**

```bash
# Stop container + remove volume (DESTROYS ALL DATA)
docker compose -f docker/docker-compose.yml down -v

# Recreate
docker compose -f docker/docker-compose.yml up -d

# Run migrations
source .venv/bin/activate
alembic -c packages/db/alembic.ini upgrade head
```

### Option B: Local PostgreSQL

If you already have PostgreSQL installed (via Homebrew, apt, etc.):

```bash
# Create the database
createdb piezo_ai
# Or:
psql -c "CREATE DATABASE piezo_ai;"

# Create a dedicated user (optional)
psql -c "CREATE USER piezo WITH PASSWORD 'piezo';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE piezo_ai TO piezo;"

# Update .env
DATABASE_URL=postgresql+asyncpg://piezo:piezo@localhost:5432/piezo_ai
```

### Running Migrations

Alembic manages schema migrations for all 6 tables:

```bash
source .venv/bin/activate

# Run pending migrations
alembic -c packages/db/alembic.ini upgrade head

# Check current migration state
alembic -c packages/db/alembic.ini current

# Create a new migration (after model changes)
alembic -c packages/db/alembic.ini revision --autogenerate -m "description"

# Rollback one migration
alembic -c packages/db/alembic.ini downgrade -1

# Full reset (drop all, recreate)
alembic -c packages/db/alembic.ini downgrade base
alembic -c packages/db/alembic.ini upgrade head
```

### Connecting with psql (Debugging)

```bash
# Docker
docker exec -it piezo-ai-db psql -U piezo -d piezo_ai

# Local
psql -U piezo -d piezo_ai

# Useful queries
SELECT count(*) FROM datasets;
SELECT id, display_name, status, total_rows FROM datasets;
SELECT target, algorithm, r2_score, rmse FROM trained_models ORDER BY r2_score DESC;
SELECT formula, d33_predicted, tc_predicted FROM predictions ORDER BY created_at DESC LIMIT 10;
```

---

## dev.sh Command Reference

The `scripts/dev.sh` script (507 lines) is the primary developer interface. It's built from modular libraries in `scripts/lib/`:

| Library | File | Handles |
|---------|------|---------|
| Colors | `_colors.sh` | Terminal color definitions |
| Python | `_python.sh` (14KB) | pyenv detection, Python install, venv creation, pip install |
| Node | `_node.sh` (7KB) | nvm detection, Node install, pnpm setup |
| Database | `_database.sh` (10KB) | Docker/local PostgreSQL detection, creation, healthcheck |
| Network | `_network.sh` (5KB) | Port checking, DNS diagnostics |

### All Commands

```bash
bash scripts/dev.sh <command>
```

| Command | Description |
|---------|-------------|
| `setup` | **Incremental setup.** Only installs what's missing or outdated. Safe to re-run. |
| `setup:all` | **Full clean setup.** Deletes `.venv/`, `node_modules/`, `.next/`, `__pycache__/` and reinstalls everything from scratch. |
| `clean` | Remove all generated files (`.venv`, `node_modules`, `.next`, `__pycache__`). Does **not** touch the database. |
| `start` | Start both FastAPI (port 8000) and Next.js (port 3000) dev servers. |
| `stop` | Gracefully shut down all running servers and free ports 3000/8000. |
| `db:create` | Create the PostgreSQL database (Docker or local). |
| `db:migrate` | Run `alembic upgrade head`. |
| `db:reset` | **DESTRUCTIVE.** Drop the database, recreate it, and run all migrations. All data is lost. |
| `diagnose` | Run full system diagnostics — checks Python, Node, pnpm, Docker, network, database connectivity, port availability. Outputs a color-coded report. |

### Flags & Options

```bash
# Use local PostgreSQL instead of Docker
bash scripts/dev.sh setup --local-db

# Force clean install even if versions match
bash scripts/dev.sh setup:all

# Verbose output (show all subprocess output)
PZ_LOG_LEVEL=DEBUG bash scripts/dev.sh start

# Custom port override (if 8000 is taken)
PORT=8001 bash scripts/dev.sh start
```

---

## Running Individual Components

### Backend Only

```bash
source .venv/bin/activate

# Development (auto-reload on file changes)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Run from: apps/api/

# Production (multiple workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Health check
curl http://localhost:8000/health
# → {"status":"ok","version":"2.2.0","database":"connected"}
```

### Frontend Only

```bash
# Development (hot reload)
cd apps/web
pnpm dev
# → http://localhost:3000

# Type checking
pnpm type-check

# Lint
pnpm lint

# Production build (for validation)
pnpm build
pnpm start
```

### ML Core (Standalone)

The ML core is a regular Python package. You can use it independently:

```python
from piezo_ml.parsers.formula_parser import FormulaParser
from piezo_ml.features.feature_engineer import FeatureEngineer
from piezo_ml.registry.element_registry import ElementRegistry

# Parse a formula
parser = FormulaParser()
result = parser.parse("BaTiO3")
print(result.elements)  # {'Ba': 1.0, 'Ti': 1.0, 'O': 3.0}

# Generate features
registry = ElementRegistry()
engineer = FeatureEngineer(registry)
features = engineer.compute_features(result.elements)
print(f"Feature vector: {len(features)} dimensions")
```

### Database Package (Standalone)

```python
from piezo_db.models import Dataset, Material, TrainedModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("postgresql://piezo:piezo@localhost:5432/piezo_ai")
with Session(engine) as session:
    datasets = session.query(Dataset).all()
    for ds in datasets:
        print(f"{ds.display_name}: {ds.total_rows} rows")
```

---

## Version Management

### Python Version (.python-version)

The file `.python-version` in the project root contains `3.13` — pyenv reads this automatically.

```bash
# Check current Python
python --version  # should be 3.13.x

# If wrong version
pyenv install 3.13.3
pyenv local 3.13.3
```

### Node Version (.nvmrc)

The file `.nvmrc` in the project root contains `20` — nvm reads this automatically.

```bash
# Check current Node
node --version  # should be v20.x

# If wrong version
nvm install 20
nvm use 20
```

### pnpm Version

The project uses pnpm 10+. The `packageManager` field in root `package.json` enforces compatibility.

```bash
# Check version
pnpm --version

# Update if needed
npm install -g pnpm@latest
```

---

**← [Developer Guide](DEVELOPER_GUIDE.md)** | **Next: [ML Pipeline →](ML_PIPELINE.md)**
