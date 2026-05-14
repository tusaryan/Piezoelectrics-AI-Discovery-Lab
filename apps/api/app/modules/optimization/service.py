"""
Optimization Service — orchestrates ML-Core calls for optimization endpoints.

DUMB PIPE: loads models from DB/filesystem, delegates to ML-Core.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from piezo_db.models import TrainedModel

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[5]


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
        """Run NSGA-II optimization using trained models as surrogates."""
        from piezo_ml.optimization import (
            NSGA2Optimizer,
            OptimizationConfig,
        )

        # Load models and metadata from DB + filesystem
        models: dict[str, Any] = {}
        feature_columns: dict[str, list[str]] = {}

        for target, model_id in model_ids.items():
            loaded = await self._load_model(model_id)
            if loaded is None:
                logger.warning(f"Model {model_id} for target {target} not found")
                continue
            models[target] = loaded["model"]
            feature_columns[target] = loaded["feature_columns"]

        if not models:
            return {
                "solutions": [],
                "convergence": [],
                "error": "No valid trained models found for the selected targets",
            }

        config = OptimizationConfig(
            model_ids=model_ids,
            objectives=objectives,
            preset=preset,
            pop_size=pop_size,
            n_generations=n_generations,
            seed=seed,
            search_elements=search_elements,
        )

        optimizer = NSGA2Optimizer()
        result = optimizer.optimize(config, models, feature_columns)

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
        metadata_path = root / db_model.artifact_dir / "metadata.json"
        feature_columns = []

        # Try model-specific metadata first
        model_file_name = Path(db_model.model_file_path).stem
        specific_meta = root / db_model.artifact_dir.replace(
            "training-artifacts", "trained-models"
        ).rsplit("/", 1)[0] if "/" in db_model.artifact_dir else root / "resources" / "trained-models"

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

        # Fallback: try artifact_dir metadata
        if not feature_columns:
            artifact_meta = root / db_model.artifact_dir / "preprocessing_log.txt" if db_model.artifact_dir else None
            # Use feature_dim to infer columns if needed
            logger.warning(f"No feature_columns found for model {model_id}, using empty list")

        return {
            "model": model,
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
