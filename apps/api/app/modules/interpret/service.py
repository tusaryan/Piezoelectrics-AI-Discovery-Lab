"""
Interpret Service — orchestrates ML-Core SHAP/PySR calls.

DUMB PIPE: loads model + data from filesystem, delegates to ML-Core.

Adds:
- File-based SHAP result caching (survives page nav + server restart)
- Background task execution via multiprocessing (killable on Ctrl+C)
- XGBoost feature alignment to fix column count mismatches
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from piezo_db.models import TrainedModel

logger = logging.getLogger(__name__)

# ── Global task registry for shutdown cleanup ──
_running_processes: dict[str, multiprocessing.Process] = {}


def get_running_processes() -> dict[str, multiprocessing.Process]:
    """Return the global process registry (used by shutdown handler)."""
    return _running_processes


def kill_all_background_tasks() -> None:
    """Kill all tracked background SHAP processes. Called on server shutdown."""
    for key, proc in list(_running_processes.items()):
        if proc.is_alive():
            logger.info(f"[Shutdown] Killing background SHAP task: {key} (PID {proc.pid})")
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
        del _running_processes[key]
    logger.info("[Shutdown] All background SHAP tasks cleaned up")


def _project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[5]


def _shap_cache_dir() -> Path:
    """Get SHAP result cache directory."""
    d = _project_root() / "resources" / "shap-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fix_julia_permissions() -> None:
    """Fix Julia cache permission issues on macOS (EPERM on ~/.julia/compiled).

    macOS TCC/Gatekeeper can restrict access to ~/.julia entirely.
    The fix is to redirect Julia's depot path to a location inside the project
    directory that we control, bypassing the security restriction.
    """
    import platform

    if platform.system() != "Darwin":
        return

    # Check if ~/.julia is accessible
    julia_default = Path.home() / ".julia"
    try:
        list(julia_default.iterdir()) if julia_default.exists() else None
        # If we can list it, no fix needed
        return
    except PermissionError:
        pass
    except OSError:
        pass

    # Redirect Julia depot to project-local directory
    local_julia = _project_root() / "resources" / ".julia-depot"
    local_julia.mkdir(parents=True, exist_ok=True)
    os.environ["JULIA_DEPOT_PATH"] = str(local_julia)
    logger.info(
        f"[PySR] ~/.julia is not accessible (macOS security restriction). "
        f"Redirecting Julia depot to {local_julia}"
    )


def _align_features(model: Any, X: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Align DataFrame columns to match what the model expects.

    Fixes the XGBoost 'chunksize * rows' crash when feature_vectors.csv
    has different columns than the model was trained on.
    """
    # Try getting expected features from metadata first
    expected_cols = metadata.get("feature_columns", [])

    # Fallback: try model attributes
    if not expected_cols:
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
            fn = model.get_booster().feature_names
            if fn:
                expected_cols = list(fn)

    if not expected_cols:
        return X  # No way to align — return as-is

    current_cols = set(X.columns)
    expected_set = set(expected_cols)

    missing = expected_set - current_cols
    extra = current_cols - expected_set

    if not missing and not extra:
        # Columns match but may be in wrong order
        if list(X.columns) != expected_cols:
            return X[expected_cols]
        return X

    if missing:
        logger.warning(
            f"[Feature Align] {len(missing)} features expected by model but missing from data: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} — filling with 0.0"
        )
        for col in missing:
            X[col] = 0.0

    if extra:
        logger.info(
            f"[Feature Align] Dropping {len(extra)} extra columns not expected by model: "
            f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        )

    return X[expected_cols]


def _load_model_and_data(
    model_row: Any,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Load model .joblib and its training feature vectors from disk."""
    import platform

    root = _project_root()

    # Load model
    model_path = Path(model_row.model_file_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    # Fix: Set n_jobs=1 for XGBoost/LightGBM to prevent macOS OpenMP crashes
    # This must be done AFTER loading, not before, as joblib restores params
    model_name = type(model).__name__.lower()
    if "xgboost" in model_name or "xgbclassifier" in model_name or "xgbregressor" in model_name:
        if hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass  # Some XGBoost versions don't support n_jobs setter
    elif "lightgbm" in model_name or "lgbclassifier" in model_name:
        if hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass

    # Load metadata to get artifact dir
    meta_path = model_path.parent / model_path.name.replace(".joblib", ".json").replace(
        "model_", "metadata_"
    )
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Load feature vectors CSV from training artifacts
    # Try model_row.artifact_dir first, then fall back to metadata source_artifact_dir
    artifact_dir = model_row.artifact_dir or metadata.get("source_artifact_dir", "")
    if artifact_dir:
        art_path = Path(artifact_dir)
        if not art_path.is_absolute():
            art_path = root / artifact_dir
        fv_path = art_path / "feature_vectors.csv"
        if fv_path.exists():
            X = pd.read_csv(fv_path)
            # Remove uid and formula columns (not features)
            drop_cols = [c for c in ["uid", "formula"] if c in X.columns]
            X = X.drop(columns=drop_cols)
            # Align features to match model expectations
            X = _align_features(model, X, metadata)
            return model, X, metadata

    # Fallback: try metadata feature_columns
    feature_cols = metadata.get("feature_columns", [])
    if feature_cols:
        logger.warning("No feature_vectors.csv found — using empty DataFrame with columns from metadata")
        X = pd.DataFrame(columns=feature_cols)
        return model, X, metadata

    raise FileNotFoundError(
        f"No training data found for model. "
        f"Artifact dir: {artifact_dir}"
    )


def _load_target_values(model_row: Any) -> np.ndarray | None:
    """Load target values from training artifacts for PySR."""
    root = _project_root()
    artifact_dir = model_row.artifact_dir
    if not artifact_dir:
        return None

    art_path = Path(artifact_dir)
    if not art_path.is_absolute():
        art_path = root / artifact_dir

    # Try parsed_compositions.csv which has the target values
    parsed_path = art_path / "parsed_compositions.csv"
    target = model_row.target
    if parsed_path.exists():
        df = pd.read_csv(parsed_path)
        if target in df.columns:
            vals = pd.to_numeric(df[target], errors="coerce").dropna()
            return vals.values

    # Try source_with_uid.csv
    source_path = art_path / "source_with_uid.csv"
    if source_path.exists():
        df = pd.read_csv(source_path)
        if target in df.columns:
            vals = pd.to_numeric(df[target], errors="coerce").dropna()
            return vals.values

    return None


# ── Background SHAP worker (runs in subprocess) ──

def _shap_beeswarm_worker(
    cache_path: str,
    status_path: str,
    model_file_path: str,
    artifact_dir: str,
    metadata_json: str,
    target: str,
    algorithm: str,
    model_id: str,
    max_samples: int,
) -> None:
    """Worker function that runs in a subprocess to compute SHAP beeswarm.

    Writes results to cache_path, status updates to status_path.
    """
    import platform
    if platform.system() == "Darwin":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    status_file = Path(status_path)

    try:
        status_file.write_text(json.dumps({"status": "loading_model"}))

        # Load model
        model = joblib.load(model_file_path)
        model_name = type(model).__name__.lower()
        if "xgb" in model_name and hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass

        # Load metadata
        metadata = json.loads(metadata_json) if metadata_json else {}

        # Load feature vectors
        root = Path(model_file_path).resolve().parents[2]  # resources/models → root
        fv_path = None
        if artifact_dir:
            art = Path(artifact_dir)
            if not art.is_absolute():
                art = root / artifact_dir
            fv_candidate = art / "feature_vectors.csv"
            if fv_candidate.exists():
                fv_path = fv_candidate

        if not fv_path:
            status_file.write_text(json.dumps({
                "status": "error",
                "error": "No feature_vectors.csv found for this model",
            }))
            return

        X = pd.read_csv(fv_path)
        drop_cols = [c for c in ["uid", "formula"] if c in X.columns]
        X = X.drop(columns=drop_cols)
        X = _align_features(model, X, metadata)

        if len(X) == 0:
            status_file.write_text(json.dumps({
                "status": "error",
                "error": "No training data available for SHAP analysis",
            }))
            return

        status_file.write_text(json.dumps({
            "status": "computing",
            "n_samples": min(len(X), max_samples),
            "n_features": len(X.columns),
        }))

        # Run SHAP
        from piezo_ml.evaluation import ShapAnalyzer
        analyzer = ShapAnalyzer()
        result = analyzer.compute_beeswarm(model, X, max_samples=max_samples)

        # Build output
        sorted_features = sorted(
            zip(result.feature_names, result.mean_abs_shap),
            key=lambda x: x[1], reverse=True,
        )
        top_features = [
            {"name": name, "mean_abs_shap": round(shap, 6), "rank": i + 1}
            for i, (name, shap) in enumerate(sorted_features[:20])
        ]

        output = {
            "model_id": model_id,
            "target": target,
            "algorithm": algorithm,
            "feature_names": result.feature_names,
            "shap_values": result.shap_values,
            "feature_values": result.feature_values,
            "base_value": result.base_value,
            "mean_abs_shap": result.mean_abs_shap,
            "top_features": top_features,
            "n_samples": len(result.shap_values),
        }

        # Save to cache
        Path(cache_path).write_text(json.dumps(output))
        status_file.write_text(json.dumps({"status": "completed"}))

    except Exception as e:
        status_file.write_text(json.dumps({
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }))


class InterpretService:
    """Service for interpretability analysis."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_model(self, model_id: str) -> Any:
        """Fetch a trained model row from DB."""
        stmt = select(TrainedModel).where(TrainedModel.id == UUID(model_id))
        result = await self.db.execute(stmt)
        model_row = result.scalar_one_or_none()
        if not model_row:
            raise ValueError(f"Model not found: {model_id}")
        return model_row

    async def get_models(self) -> list[Any]:
        """Fetch all trained models."""
        stmt = select(TrainedModel).order_by(TrainedModel.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ── SHAP Cache Methods ──

    def _cache_path(self, model_id: str, analysis: str) -> Path:
        return _shap_cache_dir() / f"{model_id}_{analysis}.json"

    def _status_path(self, model_id: str, analysis: str) -> Path:
        return _shap_cache_dir() / f"{model_id}_{analysis}_status.json"

    def _get_cached(self, model_id: str, analysis: str) -> dict | None:
        """Return cached result if exists."""
        path = self._cache_path(model_id, analysis)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _get_task_status(self, model_id: str, analysis: str) -> dict:
        """Get status of a running/completed task."""
        # Check cache first
        cached = self._get_cached(model_id, analysis)
        if cached:
            return {"status": "completed", "result": cached}

        # Check status file
        status_path = self._status_path(model_id, analysis)
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                # Verify process is still alive if status says computing
                task_key = f"{model_id}_{analysis}"
                proc = _running_processes.get(task_key)
                if status.get("status") == "computing":
                    if proc and proc.is_alive():
                        return status
                    elif proc and not proc.is_alive():
                        # Process died without writing results
                        return {
                            "status": "error",
                            "error": "SHAP computation process terminated unexpectedly. "
                                     "Try again — the model may require too much memory.",
                        }
                return status
            except (json.JSONDecodeError, OSError):
                pass

        return {"status": "not_started"}

    # ── SHAP Beeswarm (with background + cache) ──

    async def start_beeswarm_background(
        self, model_id: str, max_samples: int = 200,
    ) -> dict:
        """Start SHAP beeswarm in background process. Returns status."""
        # Check cache first
        cached = self._get_cached(model_id, "beeswarm")
        if cached:
            logger.info(f"[SHAP] Returning cached beeswarm for {model_id[:8]}…")
            return {"status": "completed", "result": cached}

        # Check if already computing
        task_key = f"{model_id}_beeswarm"
        existing = _running_processes.get(task_key)
        if existing and existing.is_alive():
            status = self._get_task_status(model_id, "beeswarm")
            logger.info(f"[SHAP] Beeswarm already computing for {model_id[:8]}…")
            return status

        # Load model info for the subprocess
        model_row = await self.get_model(model_id)
        root = _project_root()

        model_path = Path(model_row.model_file_path)
        if not model_path.is_absolute():
            model_path = root / model_path

        # Load metadata JSON
        meta_path = model_path.parent / model_path.name.replace(".joblib", ".json").replace(
            "model_", "metadata_"
        )
        metadata_json = ""
        if meta_path.exists():
            metadata_json = meta_path.read_text(encoding="utf-8")

        cache_path = str(self._cache_path(model_id, "beeswarm"))
        status_path = str(self._status_path(model_id, "beeswarm"))

        # Write initial status
        Path(status_path).write_text(json.dumps({"status": "starting"}))

        # Start subprocess
        proc = multiprocessing.Process(
            target=_shap_beeswarm_worker,
            args=(
                cache_path,
                status_path,
                str(model_path),
                model_row.artifact_dir or "",
                metadata_json,
                model_row.target,
                model_row.algorithm,
                model_id,
                max_samples,
            ),
            daemon=True,
        )
        proc.start()
        _running_processes[task_key] = proc

        logger.info(
            f"[SHAP] Started background beeswarm for {model_id[:8]}… "
            f"(PID {proc.pid}, max_samples={max_samples})"
        )

        return {"status": "computing", "pid": proc.pid}

    async def get_beeswarm_status(self, model_id: str) -> dict:
        """Poll the status of a beeswarm computation."""
        return self._get_task_status(model_id, "beeswarm")

    # ── Derived analyses (use cached beeswarm data — no separate SHAP runs) ──
    # Waterfall, Dependence, and Physics are all derived from the beeswarm
    # cache to avoid running KernelExplainer again (which takes 10+ minutes
    # for stacking models and crashes the proxy timeout).

    async def run_shap_waterfall(
        self, model_id: str, sample_index: int = 0,
    ) -> dict:
        """Derive SHAP waterfall for a single sample from cached beeswarm."""
        model_row = await self.get_model(model_id)

        # Try to use cached beeswarm data (avoids recomputing SHAP)
        cached = self._get_cached(model_id, "beeswarm")
        if cached:
            logger.info(f"[SHAP Waterfall] Deriving from cached beeswarm for {model_id[:8]}…")
            n_samples = cached["n_samples"]
            idx = min(sample_index, n_samples - 1)
            shap_row = cached["shap_values"][idx]
            feat_row = cached["feature_values"][idx]
            base_value = cached["base_value"]
            prediction = base_value + sum(shap_row)

            return {
                "model_id": model_id,
                "target": model_row.target,
                "feature_names": cached["feature_names"],
                "shap_values": shap_row,
                "feature_values": feat_row,
                "base_value": base_value,
                "prediction": round(prediction, 4),
                "sample_index": idx,
                "n_total_samples": n_samples,
            }

        # Fallback: trigger beeswarm computation first
        logger.info(f"[SHAP Waterfall] No cached beeswarm — triggering computation for {model_id[:8]}…")
        await self.start_beeswarm_background(model_id)
        raise ValueError(
            "SHAP beeswarm data is being computed in the background. "
            "Please click 'Load SHAP Beeswarm' first and wait for it to complete, "
            "then try Waterfall again."
        )

    async def run_shap_dependence(
        self, model_id: str, feature_name: str,
    ) -> dict:
        """Derive SHAP dependence for a feature from cached beeswarm."""
        model_row = await self.get_model(model_id)

        # Use cached beeswarm
        cached = self._get_cached(model_id, "beeswarm")
        if cached:
            logger.info(f"[SHAP Dependence] Deriving from cached beeswarm for {model_id[:8]}…")
            feature_names = cached["feature_names"]
            if feature_name not in feature_names:
                raise ValueError(
                    f"Feature '{feature_name}' not found. "
                    f"Available features: {', '.join(feature_names[:10])}..."
                )
            fi = feature_names.index(feature_name)

            # Extract SHAP values and feature values for this feature
            shap_col = [row[fi] for row in cached["shap_values"]]
            feat_col = [row[fi] for row in cached["feature_values"]]

            # Find best interaction feature (highest correlation)
            import numpy as np
            interaction_feature = None
            interaction_values = []
            best_corr = 0
            for j, other_name in enumerate(feature_names):
                if j == fi:
                    continue
                other_col = [row[j] for row in cached["feature_values"]]
                try:
                    corr = abs(float(np.corrcoef(shap_col, other_col)[0, 1]))
                    if corr > best_corr:
                        best_corr = corr
                        interaction_feature = other_name
                        interaction_values = other_col
                except (ValueError, IndexError):
                    continue

            return {
                "model_id": model_id,
                "target": model_row.target,
                "feature_name": feature_name,
                "feature_values": feat_col,
                "shap_values": shap_col,
                "interaction_feature": interaction_feature,
                "interaction_values": interaction_values,
            }

        # Fallback
        logger.info(f"[SHAP Dependence] No cached beeswarm — triggering for {model_id[:8]}…")
        await self.start_beeswarm_background(model_id)
        raise ValueError(
            "SHAP beeswarm data is being computed in the background. "
            "Please click 'Load SHAP Beeswarm' first and wait for it to complete, "
            "then try Dependence again."
        )

    async def run_physics_validation(self, model_id: str) -> dict:
        """Run physics validation using cached beeswarm data."""
        model_row = await self.get_model(model_id)

        # Use cached beeswarm
        cached = self._get_cached(model_id, "beeswarm")
        if not cached:
            logger.info(f"[Physics] No cached beeswarm — triggering for {model_id[:8]}…")
            await self.start_beeswarm_background(model_id)
            raise ValueError(
                "SHAP beeswarm data is being computed in the background. "
                "Please click 'Load SHAP Beeswarm' first and wait for it to complete, "
                "then try Physics Validation again."
            )

        logger.info(f"[Physics] Using cached beeswarm for {model_id[:8]}…")

        from piezo_ml.evaluation import PhysicsValidator
        from starlette.concurrency import run_in_threadpool

        validator = PhysicsValidator()
        result = await run_in_threadpool(
            validator.validate,
            feature_names=cached["feature_names"],
            mean_abs_shap=cached["mean_abs_shap"],
            shap_values_matrix=cached["shap_values"],
            target=model_row.target,
        )

        return {
            "model_id": model_id,
            "target": model_row.target,
            "alignment_score": result.alignment_score,
            "total_checks": result.total_checks,
            "confirmed": result.confirmed,
            "violations": [
                {
                    "feature": c.feature,
                    "expected_effect": c.expected_effect,
                    "physics_reason": c.physics_reason,
                    "actual_effect": c.actual_effect,
                    "aligned": c.aligned,
                    "shap_magnitude": c.shap_magnitude,
                    "shap_rank": c.shap_rank,
                }
                for c in result.violations
            ],
            "confirmed_checks": [
                {
                    "feature": c.feature,
                    "expected_effect": c.expected_effect,
                    "physics_reason": c.physics_reason,
                    "actual_effect": c.actual_effect,
                    "aligned": c.aligned,
                    "shap_magnitude": c.shap_magnitude,
                    "shap_rank": c.shap_rank,
                }
                for c in result.confirmed_checks
            ],
            "skipped": result.skipped,
        }

    async def run_symbolic_regression(
        self, model_id: str, max_complexity: int = 20,
        n_iterations: int = 40, timeout_seconds: int = 120,
    ) -> dict:
        """Run PySR symbolic regression."""
        model_row = await self.get_model(model_id)
        model, X, metadata = _load_model_and_data(model_row)

        if len(X) == 0:
            raise ValueError("No training data available for symbolic regression")

        y = _load_target_values(model_row)
        if y is None or len(y) == 0:
            raise ValueError(
                f"No target values found for '{model_row.target}' in training artifacts"
            )

        # Align X rows to y length
        if len(X) > len(y):
            X = X.iloc[:len(y)]
        elif len(y) > len(X):
            y = y[:len(X)]

        # Fix Julia cache permissions on macOS (EPERM on ~/.julia/compiled)
        _fix_julia_permissions()

        # Check PySR availability without initializing Julia
        try:
            import importlib.util
            spec = importlib.util.find_spec("pysr")
            if spec is None:
                return {
                    "model_id": model_id,
                    "target": model_row.target,
                    "equations": [],
                    "best_equation": None,
                    "pareto_front": [],
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "available": False,
                    "error": "PySR is not installed. Click 'Install PySR Backend' to set it up.",
                }
        except Exception:
            pass

        from piezo_ml.symbolic_regression import PySRRunner
        from starlette.concurrency import run_in_threadpool

        logger.info(
            f"[PySR] Starting symbolic regression for {model_row.target}: "
            f"{len(X)} samples × {len(X.columns)} features, "
            f"max_complexity={max_complexity}, n_iterations={n_iterations}"
        )

        runner = PySRRunner()

        if not runner.available:
            logger.error("[PySR] Julia/PySR backend not available")
            return {
                "model_id": model_id,
                "target": model_row.target,
                "equations": [],
                "best_equation": None,
                "pareto_front": [],
                "n_samples": len(X),
                "n_features": len(X.columns),
                "available": False,
                "error": (
                    "PySR Julia backend failed to initialize. "
                    "This is usually caused by Julia cache permission issues on macOS. "
                    "Try running in terminal: chmod -R u+rwX ~/.julia && "
                    "python -c 'import pysr; pysr.install()'"
                ),
            }

        result = await run_in_threadpool(
            runner.run,
            X, y,
            target=model_row.target,
            max_complexity=max_complexity,
            n_iterations=n_iterations,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            f"[PySR] Completed: {len(result.equations)} equations discovered, "
            f"best R²={result.best_equation.r2 if result.best_equation else 'N/A'}"
        )

        best = None
        if result.best_equation:
            best = {
                "equation_str": result.best_equation.equation_str,
                "latex": result.best_equation.latex,
                "complexity": result.best_equation.complexity,
                "loss": result.best_equation.loss,
                "r2": result.best_equation.r2,
                "readable": result.best_equation.readable,
            }

        return {
            "model_id": model_id,
            "target": model_row.target,
            "equations": [
                {
                    "equation_str": eq.equation_str,
                    "latex": eq.latex,
                    "complexity": eq.complexity,
                    "loss": eq.loss,
                    "r2": eq.r2,
                    "readable": eq.readable,
                }
                for eq in result.equations
            ],
            "best_equation": best,
            "pareto_front": result.pareto_front,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "available": result.available,
            "error": result.error,
        }

    async def install_pysr_backend(self) -> dict:
        """Run PySR installation (downloads Julia and setups dependencies).

        Uses the .venv Python if available, falling back to system python.
        Prompts user for confirmation since it downloads ~300MB of Julia binaries.
        """
        import asyncio
        import subprocess

        # Find the right Python (prefer .venv, fallback to system)
        root = _project_root()
        venv_python = root / ".venv" / "bin" / "python3"
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = "python3"

        try:
            # pysr.install() downloads Julia (~300MB) and sets up the environment
            cmd = [python_cmd, "-c", "import pysr; pysr.install(quiet=True)"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode(errors="replace") or stdout.decode(errors="replace")
                logger.error(f"PySR install failed: {err_msg}")
                raise ValueError(
                    f"PySR installation failed. Julia download may have been interrupted or failed.\n"
                    f"Error: {err_msg[:500]}\n\n"
                    f"Manual install: Run in terminal: {python_cmd} -c 'import pysr; pysr.install()'"
                )

            return {
                "success": True,
                "message": "PySR backend installed successfully. Julia binaries are now available.",
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"PySR install error: {e}", exc_info=True)
            raise ValueError(
                f"Installation command failed: {str(e)}\n\n"
                f"Manual install: bash scripts/dev.sh setup (to reinstall all deps with symbolic extras)\n"
                f"Or manually: source .venv/bin/activate && pip install 'piezo-ml[symbolic]' && python -c 'import pysr; pysr.install()'"
            )
