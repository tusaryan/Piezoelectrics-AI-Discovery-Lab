"""
SHAP Analyzer — compute SHAP values for trained models.

Provides data for:
- Beeswarm (global feature importance)
- Waterfall (local single-sample explanation)
- Dependence (feature-value vs SHAP-value scatter)

Uses TreeExplainer for tree-based models, KernelExplainer as fallback.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BeeswarmData:
    """Data for rendering a SHAP beeswarm plot."""
    feature_names: list[str]
    shap_values: list[list[float]]   # [n_samples x n_features]
    feature_values: list[list[float]] # [n_samples x n_features]
    base_value: float
    mean_abs_shap: list[float]        # mean |SHAP| per feature (for sorting)


@dataclass
class WaterfallData:
    """Data for rendering a SHAP waterfall plot for a single sample."""
    feature_names: list[str]
    shap_values: list[float]     # contribution of each feature
    feature_values: list[float]  # actual feature values for this sample
    base_value: float
    prediction: float
    sample_index: int


@dataclass
class DependenceData:
    """Data for rendering a SHAP dependence plot."""
    feature_name: str
    feature_values: list[float]         # x-axis: the feature's values
    shap_values: list[float]            # y-axis: SHAP values for this feature
    interaction_feature: str | None     # color: auto-detected interaction feature
    interaction_values: list[float]     # color values


def _safe_shap_import():
    """Import SHAP with suppressed warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import shap
    return shap


class ShapAnalyzer:
    """Compute SHAP values for trained ML models."""

    def __init__(self) -> None:
        self._shap = _safe_shap_import()

    def _create_explainer(
        self, model: Any, X: pd.DataFrame | np.ndarray,
    ) -> Any:
        """Create the best SHAP explainer for the model type."""
        shap = self._shap
        model_name = type(model).__name__

        # --- Tree-based models: use fast TreeExplainer ---
        tree_types = (
            "XGBRegressor", "XGBClassifier",
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
            "LGBMRegressor", "LGBMClassifier",
            "DecisionTreeRegressor", "DecisionTreeClassifier",
        )
        if model_name in tree_types:
            try:
                return shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"TreeExplainer failed for {model_name}: {e}")

        # --- Complex ensemble models (Stacking/Voting): skip auto, go to Kernel ---
        complex_types = (
            "StackingRegressor", "StackingClassifier",
            "VotingRegressor", "VotingClassifier",
        )
        if model_name not in complex_types:
            # Try Explainer (auto-detect) — skipped for complex ensembles
            try:
                return shap.Explainer(model, X)
            except Exception as e:
                logger.warning(f"Auto Explainer failed: {e}")

        # --- Fallback: KernelExplainer with safe predict wrapper ---
        return self._create_kernel_explainer(shap, model, X)

    def _create_kernel_explainer(
        self, shap: Any, model: Any, X: pd.DataFrame | np.ndarray,
    ) -> Any:
        """Create KernelExplainer with a safe predict wrapper."""
        model_name = type(model).__name__

        # Build safe background data (no NaN, use median imputation)
        if isinstance(X, pd.DataFrame):
            bg = X.copy()
            for col in bg.columns:
                bg[col] = pd.to_numeric(bg[col], errors="coerce")
            bg = bg.fillna(bg.median()).fillna(0)
        else:
            bg = np.nan_to_num(X.copy(), nan=0.0)

        n_bg = min(50, len(bg))
        background = shap.sample(bg, n_bg)

        # Wrap model.predict in a safe callable — always use numpy
        # arrays to avoid sklearn 'fitted without feature names' warnings
        def safe_predict(data: np.ndarray) -> np.ndarray:
            clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return model.predict(clean)
            except Exception:
                return np.zeros(len(clean))

        # Validate the safe_predict wrapper works on background
        try:
            test_pred = safe_predict(background if isinstance(background, np.ndarray) else background.values)
            if test_pred is None or len(test_pred) == 0:
                raise ValueError("Predict returned empty results")
        except Exception as e:
            raise ValueError(
                f"Cannot create SHAP explainer for {model_name}: "
                f"model.predict() fails on training data. Error: {e}"
            ) from e

        try:
            return shap.KernelExplainer(safe_predict, background)
        except Exception as e:
            raise ValueError(
                f"SHAP analysis is not supported for {model_name} models. "
                f"Try using a simpler model (RandomForest, XGBoost, GradientBoosting). "
                f"Error: {e}"
            ) from e

    def compute_beeswarm(
        self,
        model: Any,
        X: pd.DataFrame,
        max_samples: int = 200,
    ) -> BeeswarmData:
        """Compute SHAP values for beeswarm plot (global importance)."""
        # Subsample if too large
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X.copy()

        explainer = self._create_explainer(model, X_sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer(X_sample)

        # Extract values — handle different SHAP return types
        sv = self._extract_values(shap_values)
        base = self._extract_base_value(shap_values)

        feature_names = list(X_sample.columns)
        sv_list = sv.tolist()
        fv_list = X_sample.values.tolist()
        mean_abs = np.abs(sv).mean(axis=0).tolist()

        return BeeswarmData(
            feature_names=feature_names,
            shap_values=sv_list,
            feature_values=fv_list,
            base_value=float(base),
            mean_abs_shap=mean_abs,
        )

    def compute_waterfall(
        self,
        model: Any,
        X: pd.DataFrame,
        sample_index: int = 0,
    ) -> WaterfallData:
        """Compute SHAP waterfall for a single sample."""
        if sample_index < 0 or sample_index >= len(X):
            sample_index = 0

        explainer = self._create_explainer(model, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer(X)

        sv = self._extract_values(shap_values)
        base = self._extract_base_value(shap_values)

        sample_sv = sv[sample_index].tolist()
        sample_fv = X.iloc[sample_index].values.tolist()
        prediction = float(base + sum(sample_sv))

        return WaterfallData(
            feature_names=list(X.columns),
            shap_values=sample_sv,
            feature_values=sample_fv,
            base_value=float(base),
            prediction=prediction,
            sample_index=sample_index,
        )

    def compute_dependence(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_name: str,
    ) -> DependenceData:
        """Compute SHAP dependence for a specific feature."""
        if feature_name not in X.columns:
            raise ValueError(f"Feature '{feature_name}' not found. Available: {list(X.columns)}")

        explainer = self._create_explainer(model, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer(X)
        sv = self._extract_values(shap_values)

        feat_idx = list(X.columns).index(feature_name)
        feat_shap = sv[:, feat_idx]
        feat_vals = X[feature_name].values

        # Auto-detect interaction feature (highest abs correlation)
        interaction_feat, interaction_vals = self._find_interaction(
            sv, X, feat_idx,
        )

        return DependenceData(
            feature_name=feature_name,
            feature_values=feat_vals.tolist(),
            shap_values=feat_shap.tolist(),
            interaction_feature=interaction_feat,
            interaction_values=interaction_vals,
        )

    def _extract_values(self, shap_result: Any) -> np.ndarray:
        """Extract SHAP values array from different result types."""
        if hasattr(shap_result, "values"):
            vals = shap_result.values
            if isinstance(vals, np.ndarray):
                if vals.ndim == 3:
                    return vals[:, :, 0]
                return vals
        if isinstance(shap_result, np.ndarray):
            if shap_result.ndim == 3:
                return shap_result[:, :, 0]
            return shap_result
        return np.array(shap_result)

    def _extract_base_value(self, shap_result: Any) -> float:
        """Extract base value from SHAP explanation."""
        if hasattr(shap_result, "base_values"):
            bv = shap_result.base_values
            if isinstance(bv, np.ndarray):
                return float(bv.flat[0])
            return float(bv)
        return 0.0

    def _find_interaction(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        target_idx: int,
    ) -> tuple[str | None, list[float]]:
        """Find the feature with strongest interaction with target feature."""
        if shap_values.shape[1] <= 1:
            return None, []

        target_shap = shap_values[:, target_idx]
        best_corr = 0.0
        best_idx = -1

        for i in range(shap_values.shape[1]):
            if i == target_idx:
                continue
            feat_vals = X.iloc[:, i].values.astype(float)
            # Correlation between other feature values and target SHAP values
            if np.std(feat_vals) < 1e-10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = abs(np.corrcoef(feat_vals, target_shap)[0, 1])
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_idx = i

        if best_idx >= 0:
            return X.columns[best_idx], X.iloc[:, best_idx].values.tolist()
        return None, []
