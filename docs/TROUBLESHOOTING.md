# Troubleshooting

> Logging architecture, diagnostics, common errors, macOS-specific fixes, and known limitations.

**← [Back to Developer Guide](DEVELOPER_GUIDE.md)**

---

## Table of Contents

- [Logging Architecture](#logging-architecture)
- [Diagnostic Commands](#diagnostic-commands)
- [Common Errors](#common-errors)
- [macOS-Specific Issues](#macos-specific-issues)
- [Database Troubleshooting](#database-troubleshooting)
- [ML Pipeline Issues](#ml-pipeline-issues)
- [Frontend Issues](#frontend-issues)
- [Known Limitations](#known-limitations)

---

## Logging Architecture

Piezo.AI uses a **dual-output logging system** designed to keep your terminal clean while capturing everything to disk.

### Log Outputs

| Output | Level | Format | Purpose |
|--------|-------|--------|---------|
| **Terminal (stdout)** | `INFO` (default) | `HH:MM:SS LEVEL message` | Clean, essential output — startup, errors, warnings |
| **`logs/piezo-ai.log`** | `DEBUG` | `YYYY-MM-DD HH:MM:SS \| LEVEL \| module:line \| message` | Complete rotating log (10MB × 5 backups) |
| **`logs/piezo-ai-errors.log`** | `ERROR` | Same as above | Errors only (5MB × 3 backups) |
| **`logs/backend_detailed_<timestamp>.log`** | `DEBUG` | Same as above | Per-session full capture (no rotation) |

### Controlling Log Verbosity

```bash
# Default: only INFO+ in terminal
bash scripts/dev.sh start

# Verbose: show ALL logs including ML pipeline, SQL queries, SHAP output
PZ_LOG_LEVEL=DEBUG bash scripts/dev.sh start

# Quiet: warnings and errors only
PZ_LOG_LEVEL=WARNING bash scripts/dev.sh start
```

### Suppressed Third-Party Loggers

These noisy loggers are always suppressed from the terminal (captured in log files):
- `shap`, `numpy`, `sklearn`, `matplotlib`, `PIL`, `fsspec`
- `uvicorn.access`, `sqlalchemy.engine`, `xgboost`, `lightgbm`
- `piezo_ml` (the ML core itself)

To see their output, set `PZ_LOG_LEVEL=DEBUG`.

### Where to Look When Debugging

| Scenario | Look At |
|----------|---------|
| API crash or 500 error | `logs/piezo-ai-errors.log` → most recent entry |
| Training produces wrong results | `logs/backend_detailed_<latest>.log` → search for "train" |
| Formula parsing fails silently | `logs/piezo-ai.log` → search for "parse" or "skip" |
| Database connection issues | Terminal output (always shown) |
| SHAP analysis hangs | `logs/backend_detailed_<latest>.log` → search for "shap" |

---

## Diagnostic Commands

### Built-in Diagnostics

```bash
# Full system diagnostic — checks everything
bash scripts/dev.sh diagnose
```

This checks:
- ✓ Python version (3.13.x via pyenv)
- ✓ Node version (20.x via nvm)
- ✓ pnpm version
- ✓ Docker availability
- ✓ PostgreSQL connectivity
- ✓ Port 3000/8000 availability
- ✓ Network/DNS
- ✓ `.venv/` health

### Manual Health Checks

```bash
# Backend health
curl http://localhost:8000/health
# → {"status":"ok","version":"2.2.0","database":"connected"}

# Backend API docs
open http://localhost:8000/docs    # Swagger UI
open http://localhost:8000/redoc   # ReDoc

# Database
docker exec piezo-ai-db pg_isready -U piezo -d piezo_ai

# Python environment
source .venv/bin/activate
python -c "import piezo_ml; print('OK')"
python -c "import piezo_db; print('OK')"

# Node
node --version   # should be v20.x
pnpm --version   # should be 10.x+
```

---

## Common Errors

### `ModuleNotFoundError: No module named 'piezo_ml'`

**Cause:** Python packages not installed in editable mode.

```bash
source .venv/bin/activate
pip install -e packages/ml-core
pip install -e packages/db
pip install -e apps/api
```

### `StringDataRightTruncationError` during dataset mapping

**Cause:** A VARCHAR column in the database is too short for the data. Usually `formula` or a composite field.

**Fix:** Check the model definition in `packages/db/piezo_db/models.py`, increase the `String(N)` length, then create and run a migration:
```bash
alembic -c packages/db/alembic.ini revision --autogenerate -m "increase varchar length"
alembic -c packages/db/alembic.ini upgrade head
```

### `connection refused` / Backend can't connect to database

**Cause:** PostgreSQL is not running.

```bash
# Check Docker
docker ps | grep piezo-ai-db

# If not running
docker compose -f docker/docker-compose.yml up -d

# If using local PostgreSQL
pg_isready
# If not ready:
brew services start postgresql@16  # macOS
sudo systemctl start postgresql    # Linux
```

### `CORS error` in browser console

**Cause:** Frontend URL not in `CORS_ORIGINS`.

```bash
# In .env
CORS_ORIGINS=["http://localhost:3000"]
# If using a different port:
CORS_ORIGINS=["http://localhost:3000","http://localhost:3001"]
```

### Training produces `NaN` R² or `Inf` RMSE

**Cause:** Invalid data in the training set (constant columns, all-NaN target, zero variance).

**Fix:** The v2.2.0 trainer includes strict metric validation that rejects NaN/Inf models. If this persists:
1. Check your dataset for constant-value columns
2. Ensure target columns have sufficient non-null values (minimum ~20 rows)
3. Try a different missing value strategy (Mode instead of KNN)
4. Check `logs/piezo-ai-errors.log` for the specific error

### `Internal Server Error` when loading predictions

**Cause:** Corrupt model entries in database (from before v2.2.0 metric validation).

```bash
# Connect to DB and check
docker exec -it piezo-ai-db psql -U piezo -d piezo_ai
SELECT id, target, r2_score, rmse FROM trained_models WHERE r2_score IS NULL OR rmse IS NULL;
# Delete corrupt entries
DELETE FROM trained_models WHERE r2_score IS NULL OR rmse IS NULL;
```

### Port already in use

```bash
# Find what's using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>

# Or use dev.sh
bash scripts/dev.sh stop
```

---

## macOS-Specific Issues

### PySR / Julia EPERM Error

**Symptom:** `EPERM: operation not permitted` when PySR tries to compile Julia packages.

**Cause:** macOS code-signing enforcement blocks Julia binary execution.

**Fix:**
```bash
# Remove quarantine flag from Julia
xattr -d com.apple.quarantine $(which julia)
# Or from Homebrew installation
xattr -r -d com.apple.quarantine /usr/local/Cellar/julia/

# If using juliaup
xattr -r -d com.apple.quarantine ~/.juliaup/
```

If Julia is still blocked, add it to System Settings → Privacy & Security → Developer Tools.

### OpenMP / XGBoost Fork Safety

**Symptom:** `OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.`

**Cause:** macOS doesn't allow multiple OpenMP instances (happens with XGBoost/LightGBM when using `n_jobs=-1`).

**Handled by:** `packages/ml-core/piezo_ml/models/platform_utils.py` automatically sets safe `n_jobs` values on macOS. No action needed — but if you see this, set:
```bash
export OMP_NUM_THREADS=1
```

### pyenv Build Failures

**Symptom:** `pyenv install 3.13.3` fails with C compilation errors.

**Fix:** Install Xcode Command Line Tools and build dependencies:
```bash
xcode-select --install
brew install openssl readline sqlite3 xz zlib tcl-tk
```

### Homebrew PostgreSQL Conflicts

**Symptom:** Docker PostgreSQL can't bind to port 5432 because local PostgreSQL is running.

```bash
# Check
brew services list | grep postgresql
# Stop local
brew services stop postgresql@16
# Then start Docker
docker compose -f docker/docker-compose.yml up -d
```

---

## Database Troubleshooting

### Migrations Out of Sync

**Symptom:** `alembic upgrade head` fails with "Can't locate revision identified by..."

```bash
# Check current state
alembic -c packages/db/alembic.ini current

# If corrupt, stamp current and try again
alembic -c packages/db/alembic.ini stamp head

# Nuclear option: full reset (DESTROYS DATA)
bash scripts/dev.sh db:reset
```

### Cascading Delete Issues

Deleting a dataset cascades to: materials → training_jobs → trained_models. If orphaned models exist:

```sql
-- Find orphaned models (no parent dataset)
SELECT id, target, algorithm FROM trained_models
WHERE dataset_id NOT IN (SELECT id FROM datasets);

-- Clean up
DELETE FROM trained_models
WHERE dataset_id NOT IN (SELECT id FROM datasets);
```

### Large Database Size

```sql
-- Check table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- Reclaim space
VACUUM FULL;
```

---

## ML Pipeline Issues

### Feature Dimension Mismatch

**Symptom:** `ValueError: X has N features, but model expects M features` during prediction.

**Cause:** The model was trained on a dataset with different elements than the formula being predicted.

**Fix:** The model's `supported_elements` list is stored in its metadata. Predictions for formulas containing unsupported elements return `status: "unsupported_elements"` with a clear message. Train a new model on a dataset that includes the needed elements.

### SHAP Analysis Timeout

**Symptom:** SHAP analysis request returns 504 Gateway Timeout.

**Cause (v2.1.0):** SHAP ran synchronously in the request handler.

**Fix (v2.2.0):** SHAP runs as a background task with polling. If you still experience issues:
1. Check that the frontend is polling the status endpoint (not hanging on a single request)
2. Check `logs/backend_detailed_*.log` for SHAP progress
3. For large models (>100 features, >200 samples), SHAP can take 30+ seconds

### Stacking Ensemble Very Slow

**Expected behavior:** Stacking trains RF + XGBoost + SVR internally with 5-fold CV, then fits Ridge on top. This is 3–5× slower than individual algorithms.

For 200+ rows: expect 3–10 seconds. For 500+ rows: expect 10–30 seconds.

---

## Frontend Issues

### "Failed to fetch" / Backend Offline

Check:
1. Backend is running: `curl http://localhost:8000/health`
2. `.env` has correct `NEXT_PUBLIC_API_URL=http://localhost:8000`
3. No CORS issues (check browser console)

### Hot Reload Not Working

```bash
# Clear Next.js cache
rm -rf apps/web/.next
cd apps/web && pnpm dev
```

### TypeScript Errors After Package Update

```bash
cd apps/web
pnpm type-check  # see specific errors
pnpm install     # ensure deps are synced
```

---

## Known Limitations

| Limitation | Impact | Planned Fix |
|-----------|--------|-------------|
| GNN module is deferred | No crystal-structure-aware predictions | v3 roadmap |
| Agent module is deferred | No autonomous experimental planning | v3 roadmap |
| Voice module is deferred | No voice interaction | v3 roadmap |
| PySR requires Julia runtime | Not available out-of-the-box, ~500MB install | Consider bundling in Docker |
| Maximum 42 elements | Formulas with unsupported elements fail | Add via Settings → Elements |
| Batch prediction is synchronous | Large CSVs (>1000 rows) may timeout | Background task migration |
| Single-user only | No authentication or multi-user sessions | Not planned for research tool |
| No GPU acceleration | All ML runs on CPU | Sufficient for current model sizes |

---

**← [Interface Gallery](INTERFACE_GALLERY.md)** | **[Back to Developer Guide](DEVELOPER_GUIDE.md)**
