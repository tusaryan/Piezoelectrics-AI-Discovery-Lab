# Piezo.AI v2.1.0

> AI-driven discovery platform for lead-free piezoelectric materials

## Prerequisites

| Tool | Required Version | Why |
|------|-----------------|-----|
| **Python** | **3.13.x** (3.11–3.13 accepted) | `mendeleev` requires `<3.14`. Python 3.14 will **NOT** work |
| **Node.js** | **20.x** (LTS) | Next.js 15 requires Node 20+ |
| **pnpm** | 10+ | Monorepo workspace package manager |
| **Docker Desktop** | Latest | PostgreSQL container |

### Setting up the correct versions

**Python 3.13** — choose one method:

```bash
# Option A: Homebrew (macOS)
brew install python@3.13
python3.13 --version   # verify: 3.13.x

# Option B: pyenv (cross-platform — recommended)
pyenv install 3.13
pyenv local 3.13       # reads .python-version → auto-selects 3.13
python --version       # verify: 3.13.x (pyenv shim activates it)
```

> **How auto-detection works:** The `.python-version` file (containing `3.13`) tells pyenv which Python to use in this directory. When you run `python` or `python3`, pyenv's shim intercepts the command and routes it to Python 3.13. The `dev.sh setup` script also auto-detects `python3.13` → `python3.12` → `python3.11` from PATH/Homebrew, so it works even without pyenv.
>
> The `.venv/` virtual environment inherits whatever Python version created it. If you use `python3.13 -m venv .venv`, all `pip install` commands inside the venv use 3.13 — nothing is installed globally.

**Node.js 20** (via nvm):

```bash
# Install Node 20 if not already installed
nvm install 20

# Switch to Node 20 in this project
nvm use      # reads .nvmrc → auto-selects Node 20

# Verify
node --version   # Should show v20.x.x

# Install pnpm (one-time global)
npm install -g pnpm
```

> **How auto-detection works:** The `.nvmrc` file (containing `20`) tells nvm which Node to use. Running `nvm use` in the project root automatically switches to Node 20.

## Quick Start

```bash
# 1. Set correct Node version
nvm use

# 2. Full setup (installs everything + starts DB)
bash scripts/dev.sh setup

# 3. Start development servers
bash scripts/dev.sh start
```

This will:
1. Create a Python virtual environment (`.venv/`) using Python 3.13
2. Install all Python packages in the venv (not globally)
3. Install frontend dependencies via pnpm
4. Start PostgreSQL via Docker
5. Run database migrations
6. Start FastAPI backend on `http://localhost:8000`
7. Start Next.js frontend on `http://localhost:3000`

## Development Commands

```bash
bash scripts/dev.sh <command>
```

| Command | Description |
|---------|-------------|
| `setup` | Incremental setup (keeps existing deps if compatible) |
| `setup:all` | Full clean + fresh install (wipes .venv, node_modules) |
| `clean` | Remove node_modules, .next, __pycache__, .venv |
| `db:create` | Create the PostgreSQL database |
| `db:reset` | Drop and recreate DB + run migrations (**destroys data**) |
| `db:migrate` | Run Alembic migrations only |
| `start` | Start backend + frontend dev servers |
| `stop` | Gracefully shut down all servers + free ports |

## Alembic Migration Guide

Run migrations from `packages/db`:

```bash
cd packages/db
alembic upgrade head
```

If you prefer explicit config path from repo root:

```bash
alembic -c packages/db/alembic.ini upgrade head
```

If you are already inside `packages/db`, use:

```bash
alembic -c alembic.ini upgrade head
```

Common pitfall:
- `alembic -c packages/db/alembic.ini upgrade head` fails when run *inside* `packages/db` because that path becomes `packages/db/packages/db/alembic.ini`.

## Manual Setup (if dev.sh doesn't work)

```bash
# 1. Set versions
nvm use 20
# Python 3.13 is auto-detected by dev.sh, or manually:

# 2. Create virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate

# 3. Install Python packages (in venv — nothing global)
pip install -e packages/db
pip install -e packages/ml-core
pip install -e apps/api

# 4. Start PostgreSQL
docker compose -f docker/docker-compose.yml up -d

# 5. Start backend
cd apps/api && uvicorn app.main:app --reload --port 8000

# 6. Start frontend (new terminal)
cd /path/to/project
nvm use
pnpm install
pnpm dev:web
```

## Project Structure

```
├── apps/
│   ├── api/          # FastAPI backend (DUMB PIPE — no ML logic)
│   └── web/          # Next.js 15 + React 19 frontend
├── packages/
│   ├── ml-core/      # ALL ML logic (registry, parsers, training, prediction)
│   └── db/           # SQLAlchemy models + Alembic migrations
├── resources/
│   ├── main-datasets/          # Source datasets
│   ├── sample-and-test-dataset/ # Test datasets
│   ├── training-artifacts/     # Parsed datasets per training run
│   └── trained-models/         # Saved .joblib models
├── scripts/          # dev.sh
├── docker/           # Docker Compose for PostgreSQL
└── Project/          # Implementation plans & session tracker
```

## Architecture

- **FastAPI** is a dumb pipe — all ML logic lives in `packages/ml-core/`
- **Central Element Registry** — single source of truth for 33 supported elements
- **3 Themes** — Dark (default), Light, Night (warm amber)
- **7 Sections** — Dashboard, Dataset, Train, Predict, Optimization Lab, Interpretability, Settings
- **Virtual environment** — all Python deps installed in `.venv/`, nothing pollutes your system

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, TailwindCSS 4, Framer Motion |
| Backend | FastAPI, Uvicorn, SQLAlchemy (async) |
| Database | PostgreSQL 16 |
| ML | scikit-learn, XGBoost, LightGBM, SHAP, Optuna |
| Chemistry | chemparse, pymatgen, mendeleev |
| Python | 3.13 (venv-isolated) |
| Node.js | 20 LTS (via nvm) |

## Troubleshooting

**`mendeleev` / `pymatgen` install fails:**
→ You're using Python 3.14. Switch to 3.13: `python3.13 -m venv .venv`

**`pnpm: command not found`:**
→ `npm install -g pnpm`

**Port 8000/3000 already in use:**
→ `bash scripts/dev.sh stop` or manually: `lsof -ti:8000 | xargs kill -9`

**Docker permission denied:**
→ Make sure Docker Desktop is running

## License

This project is part of academic research.
