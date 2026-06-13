"""
Optimization Service — orchestrates ML-Core calls for optimization endpoints.

DUMB PIPE: loads models from DB/filesystem, delegates to ML-Core.

Adds:
- Background NSGA-II execution via multiprocessing (killable on Ctrl+C)
- File-based result caching (survives proxy timeout + page navigation)
- Process registry for shutdown cleanup
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import joblib
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
    """Kill all tracked background optimization processes. Called on server shutdown."""
    for key, proc in list(_running_processes.items()):
        if proc.is_alive():
            logger.info(f"[Shutdown] Killing background optimization task: {key} (PID {proc.pid})")
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
        del _running_processes[key]
    logger.info("[Shutdown] All background optimization tasks cleaned up")


def _project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[5]


def _optim_cache_dir() -> Path:
    """Get optimization result cache directory."""
    d = _project_root() / "resources" / "optimization-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_task_key(model_ids: dict[str, str], preset: str, pop_size: int, n_gen: int) -> str:
    """Create a unique task key from optimization parameters."""
    # Hash the model IDs + params to create a unique key
    raw = json.dumps({"m": model_ids, "p": preset, "ps": pop_size, "ng": n_gen}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Background optimization worker (runs in subprocess) ──

def _optimization_worker(
    cache_path: str,
    status_path: str,
    model_paths: dict[str, str],      # target -> absolute model path
    feature_columns: dict[str, list[str]],  # target -> feature columns
    objectives: dict[str, dict[str, Any]],
    preset: str,
    pop_size: int,
    n_generations: int,
    seed: int,
    search_elements: list[str] | None,
    targets_optimized: list[str],
) -> None:
    """Worker function that runs NSGA-II in a subprocess."""
    import platform
    if platform.system() == "Darwin":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    status_file = Path(status_path)

    try:
        status_file.write_text(json.dumps({"status": "loading_models"}))

        # Load models from disk
        models: dict[str, Any] = {}
        for target, model_path in model_paths.items():
            try:
                model = joblib.load(model_path)
                models[target] = model
            except Exception as e:
                status_file.write_text(json.dumps({
                    "status": "error",
                    "error": f"Failed to load model for '{target}': {e}",
                }))
                return

        status_file.write_text(json.dumps({
            "status": "optimizing",
            "targets": list(models.keys()),
            "pop_size": pop_size,
            "n_generations": n_generations,
        }))

        # Run NSGA-II
        from piezo_ml.optimization import NSGA2Optimizer, OptimizationConfig

        config = OptimizationConfig(
            model_ids={t: "" for t in models},  # IDs not needed for actual computation
            objectives=objectives,
            preset=preset,
            pop_size=pop_size,
            n_generations=n_generations,
            seed=seed,
            search_elements=search_elements,
        )

        optimizer = NSGA2Optimizer()
        result = optimizer.optimize(config, models, feature_columns)

        # Build output
        output = {
            "solutions": [
                {
                    "composition": s.composition,
                    "formula_approx": s.formula_approx,
                    "predicted": s.predicted,
                    "use_case_tag": s.use_case_tag,
                    "use_case_color": s.use_case_color,
                    "rank": s.rank,
                    "crowding_distance": s.crowding_distance,
                }
                for s in result.solutions
            ],
            "convergence": result.convergence,
            "n_generations_run": result.n_generations_run,
            "n_evaluations": result.n_evaluations,
            "duration_seconds": result.duration_seconds,
            "targets_optimized": result.targets_optimized,
            "preset_used": result.preset_used,
            "error": result.error,
        }

        # Save to cache
        Path(cache_path).write_text(json.dumps(output))
        status_file.write_text(json.dumps({"status": "completed"}))

    except MemoryError:
        status_file.write_text(json.dumps({
            "status": "error",
            "error": (
                "Server ran out of memory during optimization. "
                f"Try reducing Population Size (currently {pop_size}) or "
                f"Generations (currently {n_generations})."
            ),
        }))
    except Exception as e:
        status_file.write_text(json.dumps({
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
        }))


class OptimizationService:
    """Service layer for structural analysis and NSGA-II optimization."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_models(self) -> list[TrainedModel]:
        """Get all trained models available for optimization."""
        result = await self.db.execute(
            select(TrainedModel).order_by(TrainedModel.target, TrainedModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def run_structural_analysis(self, formula: str) -> dict[str, Any]:
        """Run structural analysis on a single formula."""
        from piezo_ml.optimization import StructuralAnalyzer

        analyzer = StructuralAnalyzer()
        desc = analyzer.analyze(formula)
        return _descriptor_to_dict(desc)

    async def run_structural_comparison(self, formulas: list[str]) -> list[dict[str, Any]]:
        """Analyze and compare multiple formulas."""
        from piezo_ml.optimization import StructuralAnalyzer

        analyzer = StructuralAnalyzer()
        results = analyzer.compare(formulas)
        return [_descriptor_to_dict(d) for d in results]

    # ── Optimization Cache Methods ──

    def _cache_path(self, task_key: str) -> Path:
        return _optim_cache_dir() / f"{task_key}_result.json"

    def _status_path(self, task_key: str) -> Path:
        return _optim_cache_dir() / f"{task_key}_status.json"

    def _get_cached(self, task_key: str) -> dict | None:
        """Return cached result if exists."""
        path = self._cache_path(task_key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def get_task_status(self, task_key: str) -> dict:
        """Get status of a running/completed task."""
        # Check cache first
        cached = self._get_cached(task_key)
        if cached:
            return {"status": "completed", "result": cached}

        # Check status file
        status_path = self._status_path(task_key)
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                # Verify process is still alive if status says optimizing
                proc = _running_processes.get(task_key)
                if status.get("status") == "optimizing":
                    if proc and proc.is_alive():
                        return status
                    elif proc and not proc.is_alive():
                        return {
                            "status": "error",
                            "error": "Optimization process terminated unexpectedly. Try again.",
                        }
                return status
            except (json.JSONDecodeError, OSError):
                pass

        return {"status": "not_started"}

    # ── Background Optimization ──

    async def start_optimization_background(
        self,
        model_ids: dict[str, str],
        objectives: dict[str, dict[str, Any]],
        preset: str = "custom",
        pop_size: int = 100,
        n_generations: int = 50,
        seed: int = 42,
        search_elements: list[str] | None = None,
    ) -> dict:
        """Start NSGA-II in background process. Returns status."""
        task_key = _make_task_key(model_ids, preset, pop_size, n_generations)

        # Check cache first
        cached = self._get_cached(task_key)
        if cached:
            logger.info(f"[Optimize] Returning cached result (task={task_key})")
            return {"status": "completed", "result": cached, "task_key": task_key}

        # Check if already running
        existing = _running_processes.get(task_key)
        if existing and existing.is_alive():
            status = self.get_task_status(task_key)
            logger.info(f"[Optimize] Already computing (task={task_key})")
            return {**status, "task_key": task_key}

        # Load models and resolve paths from DB
        root = _project_root()
        model_paths: dict[str, str] = {}
        feature_columns: dict[str, list[str]] = {}
        load_errors: list[str] = []

        for target, model_id in model_ids.items():
            logger.info(f"[Optimize] Loading model for target='{target}', model_id='{model_id}'")
            loaded = await self._load_model(model_id)
            if loaded is None:
                msg = f"Model '{model_id[:8]}…' for target '{target}' could not be loaded"
                logger.warning(f"[Optimize] {msg}")
                load_errors.append(msg)
                continue
            model_paths[target] = loaded["model_path"]
            feature_columns[target] = loaded["feature_columns"]
            logger.info(
                f"[Optimize] Loaded model for '{target}': "
                f"algorithm={loaded['algorithm']}, features={len(loaded['feature_columns'])}"
            )

        if not model_paths:
            error_detail = (
                "No valid trained models could be loaded. "
                + (" Issues: " + "; ".join(load_errors) if load_errors else "")
            )
            return {"status": "error", "error": error_detail, "task_key": task_key}

        # Filter objectives to active targets
        active_objectives = {t: objectives[t] for t in model_paths if t in objectives}
        if not active_objectives:
            active_objectives = {
                t: {"direction": "maximize", "min": 0, "max": 1000, "weight": 1.0}
                for t in model_paths
            }

        cache_path = str(self._cache_path(task_key))
        status_path = str(self._status_path(task_key))

        # Write initial status
        Path(status_path).write_text(json.dumps({"status": "starting"}))

        # Start subprocess
        proc = multiprocessing.Process(
            target=_optimization_worker,
            args=(
                cache_path,
                status_path,
                model_paths,
                feature_columns,
                active_objectives,
                preset,
                pop_size,
                n_generations,
                seed,
                search_elements,
                list(model_paths.keys()),
            ),
            daemon=True,
        )
        proc.start()
        _running_processes[task_key] = proc

        logger.info(
            f"[Optimize] Started background NSGA-II (task={task_key}, PID={proc.pid}, "
            f"targets={list(model_paths.keys())}, pop={pop_size}, gen={n_generations})"
        )

        return {"status": "computing", "task_key": task_key, "pid": proc.pid}

    # ── Legacy synchronous method (kept for fallback) ──

    async def run_optimization(
        self,
        model_ids: dict[str, str],
        objectives: dict[str, dict[str, Any]],
        preset: str = "custom",
        pop_size: int = 100,
        n_generations: int = 50,
        seed: int = 42,
        search_elements: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run NSGA-II optimization using trained models as surrogates (synchronous)."""
        from piezo_ml.optimization import (
            NSGA2Optimizer,
            OptimizationConfig,
        )

        # Load models and metadata from DB + filesystem
        models: dict[str, Any] = {}
        feature_columns: dict[str, list[str]] = {}
        load_errors: list[str] = []

        for target, model_id in model_ids.items():
            logger.info(f"[Optimize] Loading model for target='{target}', model_id='{model_id}'")
            loaded = await self._load_model(model_id)
            if loaded is None:
                msg = f"Model '{model_id[:8]}…' for target '{target}' could not be loaded (not found or corrupt file)"
                logger.warning(f"[Optimize] {msg}")
                load_errors.append(msg)
                continue
            if not loaded.get("feature_columns"):
                logger.warning(
                    f"[Optimize] Model '{model_id[:8]}…' for '{target}' has no feature columns — "
                    f"predictions may be inaccurate (using empty feature list)"
                )
            models[target] = loaded["model_obj"]
            feature_columns[target] = loaded["feature_columns"]
            logger.info(
                f"[Optimize] Loaded model for '{target}': "
                f"algorithm={loaded['algorithm']}, features={len(loaded['feature_columns'])}"
            )

        if not models:
            error_detail = (
                "No valid trained models could be loaded for the selected targets. "
                + (
                    " Issues: " + "; ".join(load_errors)
                    if load_errors
                    else "Ensure models are trained and their files exist on disk."
                )
            )
            logger.error(f"[Optimize] {error_detail}")
            return {
                "solutions": [],
                "convergence": [],
                "n_generations_run": 0,
                "n_evaluations": 0,
                "duration_seconds": 0.0,
                "targets_optimized": [],
                "preset_used": preset,
                "error": error_detail,
            }

        # Build config
        active_objectives = {t: objectives[t] for t in models if t in objectives}
        if not active_objectives:
            active_objectives = {
                t: {"direction": "maximize", "min": 0, "max": 1000, "weight": 1.0}
                for t in models
            }

        config = OptimizationConfig(
            model_ids={t: model_ids[t] for t in models},
            objectives=active_objectives,
            preset=preset,
            pop_size=pop_size,
            n_generations=n_generations,
            seed=seed,
            search_elements=search_elements,
        )

        from starlette.concurrency import run_in_threadpool
        optimizer = NSGA2Optimizer()

        try:
            result = await run_in_threadpool(optimizer.optimize, config, models, feature_columns)
        except Exception as e:
            logger.error(f"[Optimize] NSGA-II threadpool execution failed: {e}", exc_info=True)
            return {
                "solutions": [],
                "convergence": [],
                "n_generations_run": 0,
                "n_evaluations": 0,
                "duration_seconds": 0.0,
                "targets_optimized": list(models.keys()),
                "preset_used": preset,
                "error": f"NSGA-II computation failed: {type(e).__name__}: {str(e)}",
            }

        return {
            "solutions": [
                {
                    "composition": s.composition,
                    "formula_approx": s.formula_approx,
                    "predicted": s.predicted,
                    "use_case_tag": s.use_case_tag,
                    "use_case_color": s.use_case_color,
                    "rank": s.rank,
                    "crowding_distance": s.crowding_distance,
                }
                for s in result.solutions
            ],
            "convergence": result.convergence,
            "n_generations_run": result.n_generations_run,
            "n_evaluations": result.n_evaluations,
            "duration_seconds": result.duration_seconds,
            "targets_optimized": result.targets_optimized,
            "preset_used": result.preset_used,
            "error": result.error,
        }

    async def _load_model(self, model_id: str) -> dict[str, Any] | None:
        """Load a trained model from DB + filesystem."""
        result = await self.db.execute(
            select(TrainedModel).where(TrainedModel.id == model_id)
        )
        db_model = result.scalar_one_or_none()
        if db_model is None:
            return None

        # Resolve model file path
        root = _project_root()
        model_path = root / db_model.model_file_path
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        # Load model
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None

        # Load metadata for feature columns
        feature_columns = []

        # Search for metadata JSON in trained-models directory
        trained_dir = root / "resources" / "trained-models"
        for meta_file in trained_dir.glob("metadata_*.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                model_files = meta.get("model_files", [])
                # Check if this metadata belongs to the right model
                if any(db_model.model_file_path.endswith(mf) for mf in model_files):
                    feature_columns = meta.get("feature_columns", [])
                    break
            except Exception:
                continue

        # Fallback warning
        if not feature_columns:
            logger.warning(f"No feature_columns found for model {model_id}, using empty list")

        return {
            "model_obj": model,
            "model_path": str(model_path),
            "feature_columns": feature_columns,
            "target": db_model.target,
            "algorithm": db_model.algorithm,
        }


def _descriptor_to_dict(desc) -> dict[str, Any]:
    """Convert StructuralDescriptor dataclass to dict."""
    return {
        "formula": desc.formula,
        "normalized_formula": desc.normalized_formula,
        "is_valid": desc.is_valid,
        "error": desc.error,
        "tolerance_factor": desc.tolerance_factor,
        "octahedral_factor": desc.octahedral_factor,
        "crystal_system": desc.crystal_system,
        "stability_class": desc.stability_class,
        "avg_bond_valence_a": desc.avg_bond_valence_a,
        "avg_bond_valence_b": desc.avg_bond_valence_b,
        "bond_valence_mismatch": desc.bond_valence_mismatch,
        "a_site_elements": desc.a_site_elements,
        "b_site_elements": desc.b_site_elements,
        "dopant_elements": desc.dopant_elements,
        "oxygen_content": desc.oxygen_content,
        "total_elements": desc.total_elements,
        "avg_electronegativity": desc.avg_electronegativity,
        "electronegativity_diff": desc.electronegativity_diff,
        "avg_atomic_mass": desc.avg_atomic_mass,
        "avg_ionic_radius_a": desc.avg_ionic_radius_a,
        "avg_ionic_radius_b": desc.avg_ionic_radius_b,
        "polarizability_index": desc.polarizability_index,
        "a_site_variance": desc.a_site_variance,
        "b_site_variance": desc.b_site_variance,
        "is_perovskite_likely": desc.is_perovskite_likely,
        "perovskite_confidence": desc.perovskite_confidence,
        "phase_count": desc.phase_count,
        "warnings": desc.warnings,
    }
