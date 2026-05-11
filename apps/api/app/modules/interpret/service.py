"""
Interpret Service — orchestrates ML-Core SHAP/PySR calls.

DUMB PIPE: loads model + data from filesystem, delegates to ML-Core.
"""

from __future__ import annotations

import json
import logging
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


def _project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[5]


def _load_model_and_data(
    model_row: Any,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Load model .joblib and its training feature vectors from disk."""
    root = _project_root()

    # Load model
    model_path = Path(model_row.model_file_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    # Load metadata to get artifact dir
    meta_path = model_path.parent / model_path.name.replace(".joblib", ".json").replace(
        "model_", "metadata_"
    )
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Load feature vectors CSV from training artifacts
    artifact_dir = model_row.artifact_dir
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

    async def run_shap_beeswarm(self, model_id: str, max_samples: int = 200) -> dict:
        """Run SHAP beeswarm analysis."""
        model_row = await self.get_model(model_id)
        model, X, metadata = _load_model_and_data(model_row)

        if len(X) == 0:
            raise ValueError("No training data available for SHAP analysis")

        from piezo_ml.evaluation import ShapAnalyzer
        analyzer = ShapAnalyzer()
        result = analyzer.compute_beeswarm(model, X, max_samples=max_samples)

        # Build top features list
        sorted_features = sorted(
            zip(result.feature_names, result.mean_abs_shap),
            key=lambda x: x[1], reverse=True,
        )
        top_features = [
            {"name": name, "mean_abs_shap": round(shap, 6), "rank": i + 1}
            for i, (name, shap) in enumerate(sorted_features[:20])
        ]

        return {
            "model_id": model_id,
            "target": model_row.target,
            "algorithm": model_row.algorithm,
            "feature_names": result.feature_names,
            "shap_values": result.shap_values,
            "feature_values": result.feature_values,
            "base_value": result.base_value,
            "mean_abs_shap": result.mean_abs_shap,
            "top_features": top_features,
            "n_samples": len(result.shap_values),
        }

    async def run_shap_waterfall(
        self, model_id: str, sample_index: int = 0,
    ) -> dict:
        """Run SHAP waterfall for a single sample."""
        model_row = await self.get_model(model_id)
        model, X, metadata = _load_model_and_data(model_row)

        if len(X) == 0:
            raise ValueError("No training data available for SHAP analysis")

        from piezo_ml.evaluation import ShapAnalyzer
        analyzer = ShapAnalyzer()
        result = analyzer.compute_waterfall(model, X, sample_index=sample_index)

        return {
            "model_id": model_id,
            "target": model_row.target,
            "feature_names": result.feature_names,
            "shap_values": result.shap_values,
            "feature_values": result.feature_values,
            "base_value": result.base_value,
            "prediction": result.prediction,
            "sample_index": result.sample_index,
            "n_total_samples": len(X),
        }

    async def run_shap_dependence(
        self, model_id: str, feature_name: str,
    ) -> dict:
        """Run SHAP dependence plot for a specific feature."""
        model_row = await self.get_model(model_id)
        model, X, metadata = _load_model_and_data(model_row)

        if len(X) == 0:
            raise ValueError("No training data available for SHAP analysis")

        from piezo_ml.evaluation import ShapAnalyzer
        analyzer = ShapAnalyzer()
        result = analyzer.compute_dependence(model, X, feature_name)

        return {
            "model_id": model_id,
            "target": model_row.target,
            "feature_name": result.feature_name,
            "feature_values": result.feature_values,
            "shap_values": result.shap_values,
            "interaction_feature": result.interaction_feature,
            "interaction_values": result.interaction_values,
        }

    async def run_physics_validation(self, model_id: str) -> dict:
        """Run physics validation on SHAP data."""
        model_row = await self.get_model(model_id)
        model, X, metadata = _load_model_and_data(model_row)

        if len(X) == 0:
            raise ValueError("No training data available for physics validation")

        from piezo_ml.evaluation import ShapAnalyzer, PhysicsValidator
        analyzer = ShapAnalyzer()
        beeswarm = analyzer.compute_beeswarm(model, X, max_samples=200)

        validator = PhysicsValidator()
        result = validator.validate(
            feature_names=beeswarm.feature_names,
            mean_abs_shap=beeswarm.mean_abs_shap,
            shap_values_matrix=beeswarm.shap_values,
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

        from piezo_ml.symbolic_regression import PySRRunner
        runner = PySRRunner()
        result = runner.run(
            X, y,
            target=model_row.target,
            max_complexity=max_complexity,
            n_iterations=n_iterations,
            timeout_seconds=timeout_seconds,
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
        """Run PySR installation (downloads Julia and setups dependencies)."""
        import asyncio
        import subprocess
        
        try:
            cmd = ["python", "-c", "import pysr; pysr.install(quiet=True)"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise ValueError(f"Failed to install PySR backend: {stderr.decode()}")
                
            return {"success": True, "message": "PySR backend installed successfully."}
        except Exception as e:
            logger.error(f"PySR install error: {e}", exc_info=True)
            raise ValueError(f"Installation failed: {str(e)}")

