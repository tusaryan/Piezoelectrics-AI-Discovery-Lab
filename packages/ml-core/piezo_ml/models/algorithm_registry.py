"""
Algorithm Registry — metadata + factory for all 8 supported ML algorithms.

Each entry has: display name, hyperparameter definitions (type, min, max,
default, step, description, impact, recommended), and a build_model factory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from piezo_ml.models.platform_utils import get_safe_n_jobs

# ---------------------------------------------------------------------------
# Global environment limits (configurable via Settings UI / .env)
# ---------------------------------------------------------------------------
from pathlib import Path

def reload_registry_limits() -> None:
    """Dynamically re-read limits from .env and update ALGORITHM_REGISTRY parameter constraints."""
    current_dir = Path(__file__).resolve().parent
    env_path = None
    # Search up to 5 directories up for the .env file
    for _ in range(5):
        if (current_dir / ".env").exists():
            env_path = current_dir / ".env"
            break
        current_dir = current_dir.parent

    if env_path:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip("'\"")

    def _get_max(env_key: str, default: float) -> float:
        try:
            val = os.environ.get(env_key)
            if val is not None and val != "":
                return float(val)
        except ValueError:
            pass
        return float(default)

    max_trees = int(_get_max("ML_MAX_TREES", 3000))
    max_svr_c = float(_get_max("ML_MAX_SVR_C", 4000.0))
    max_ann_iter = int(_get_max("ML_MAX_ANN_ITER", 4000))
    max_depth = int(_get_max("ML_MAX_DEPTH", 100))

    # Dynamically update the in-memory parameter boundaries
    _XGBOOST_PARAMS["n_estimators"].max_val = max_trees
    _XGBOOST_PARAMS["max_depth"].max_val = max_depth
    _RF_PARAMS["n_estimators"].max_val = max_trees
    _RF_PARAMS["max_depth"].max_val = max_depth
    _SVR_PARAMS["C"].max_val = max_svr_c
    _LIGHTGBM_PARAMS["n_estimators"].max_val = max_trees
    _LIGHTGBM_PARAMS["max_depth"].max_val = max_depth
    _GBR_PARAMS["n_estimators"].max_val = max_trees
    _GBR_PARAMS["max_depth"].max_val = max_depth
    _DT_PARAMS["max_depth"].max_val = max_depth
    _ANN_PARAMS["max_iter"].max_val = max_ann_iter
    _STACKING_PARAMS["rf_n_estimators"].max_val = max_trees
    _STACKING_PARAMS["xgb_n_estimators"].max_val = max_trees
    _STACKING_PARAMS["svr_C"].max_val = max_svr_c
    _STACKING_PARAMS["final_reg"].max_val = max_svr_c

# Initialize once on load so the initial max_vals are 3000, 4000.0, etc.
MAX_TREES = 3000
MAX_SVR_C = 4000.0
MAX_ANN_ITER = 4000
MAX_DEPTH = 100

# ---------------------------------------------------------------------------
# Hyperparameter definition
# ---------------------------------------------------------------------------

@dataclass
class HyperparamDef:
    """Describes a single hyperparameter with its valid range and guidance."""
    name: str
    type: str  # "int" | "float" | "select"
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    default: Any = None
    options: list[str] | None = None  # for "select" type
    description: str = ""
    impact: str = ""
    recommended: Any = None


@dataclass
class AlgorithmMeta:
    """Metadata for one ML algorithm."""
    key: str
    display_name: str
    description: str
    supports_convergence: bool
    hyperparameters: dict[str, HyperparamDef] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hyperparameter definitions per algorithm
# ---------------------------------------------------------------------------

_XGBOOST_PARAMS = {
    "n_estimators": HyperparamDef(
        "n_estimators", "int", 10, MAX_TREES, 10, 100,
        description="Number of boosting rounds (trees).",
        impact="More trees → more robust but slower. Risk of overfitting if max_depth is also high.",
        recommended=100,
    ),
    "max_depth": HyperparamDef(
        "max_depth", "int", 1, MAX_DEPTH, 1, 6,
        description="Maximum tree depth per round.",
        impact="Deeper trees capture complex patterns but increase overfitting risk.",
        recommended=6,
    ),
    "learning_rate": HyperparamDef(
        "learning_rate", "float", 0.0001, 2.0, 0.01, 0.1,
        description="Step size shrinkage to prevent overfitting.",
        impact="Lower values need more trees but generalize better.",
        recommended=0.1,
    ),
    "subsample": HyperparamDef(
        "subsample", "float", 0.1, 1.0, 0.05, 0.8,
        description="Fraction of samples used per tree.",
        impact="Lower values add randomness, reducing overfitting.",
        recommended=0.8,
    ),
    "colsample_bytree": HyperparamDef(
        "colsample_bytree", "float", 0.1, 1.0, 0.05, 0.8,
        description="Fraction of features used per tree.",
        impact="Lower values increase diversity between trees.",
        recommended=0.8,
    ),
    "reg_alpha": HyperparamDef(
        "reg_alpha", "float", 0.0, 100.0, 0.1, 0.0,
        description="L1 regularization (Lasso). Encourages sparsity.",
        impact="Higher values push less-important feature weights to zero.",
        recommended=0.0,
    ),
    "reg_lambda": HyperparamDef(
        "reg_lambda", "float", 0.0, 100.0, 0.1, 1.0,
        description="L2 regularization (Ridge). Smooths weights.",
        impact="Higher values reduce model complexity.",
        recommended=1.0,
    ),
}

_RF_PARAMS = {
    "n_estimators": HyperparamDef(
        "n_estimators", "int", 10, MAX_TREES, 10, 100,
        description="Number of trees in the forest.",
        impact="More trees → more stable predictions but slower training.",
        recommended=100,
    ),
    "max_depth": HyperparamDef(
        "max_depth", "int", 1, MAX_DEPTH, 1, 10,
        description="Maximum depth of each tree. None = expand until leaves are pure.",
        impact="Deeper trees capture complex patterns but risk overfitting.",
        recommended=10,
    ),
    "min_samples_split": HyperparamDef(
        "min_samples_split", "int", 2, 20, 1, 2,
        description="Minimum samples required to split an internal node.",
        impact="Higher values prevent the tree from learning overly specific patterns.",
        recommended=2,
    ),
    "min_samples_leaf": HyperparamDef(
        "min_samples_leaf", "int", 1, 20, 1, 1,
        description="Minimum samples required at a leaf node.",
        impact="Higher values create smoother predictions.",
        recommended=1,
    ),
    "max_features": HyperparamDef(
        "max_features", "select", options=["sqrt", "log2", "0.5", "0.8", "1.0"],
        default="sqrt",
        description="Number of features to consider for best split.",
        impact="Lower fractions increase diversity between trees.",
        recommended="sqrt",
    ),
}

_SVR_PARAMS = {
    "C": HyperparamDef(
        "C", "float", 0.001, MAX_SVR_C, 0.1, 1.0,
        description="Regularization parameter.",
        impact="Higher C → less regularization, fits training data more closely.",
        recommended=1.0,
    ),
    "epsilon": HyperparamDef(
        "epsilon", "float", 0.0001, 5.0, 0.01, 0.1,
        description="Epsilon-tube within which no penalty is applied.",
        impact="Larger epsilon → simpler model, ignoring small errors.",
        recommended=0.1,
    ),
    "kernel": HyperparamDef(
        "kernel", "select", options=["rbf", "linear", "poly"],
        default="rbf",
        description="Kernel type for SVR.",
        impact="RBF handles non-linear data. Linear is faster for high-dim data.",
        recommended="rbf",
    ),
    "gamma": HyperparamDef(
        "gamma", "select", options=["scale", "auto"],
        default="scale",
        description="Kernel coefficient. 'scale' uses 1/(n_features * X.var()).",
        impact="Higher gamma → decision boundary is more local/complex.",
        recommended="scale",
    ),
}

_LIGHTGBM_PARAMS = {
    "n_estimators": HyperparamDef(
        "n_estimators", "int", 10, MAX_TREES, 10, 100,
        description="Number of boosting iterations.",
        impact="More iterations → better fit but slower and risk overfitting.",
        recommended=100,
    ),
    "max_depth": HyperparamDef(
        "max_depth", "int", -1, MAX_DEPTH, 1, -1,
        description="Maximum tree depth. -1 means no limit.",
        impact="Deeper trees capture complex patterns.",
        recommended=-1,
    ),
    "learning_rate": HyperparamDef(
        "learning_rate", "float", 0.0001, 2.0, 0.01, 0.1,
        description="Boosting learning rate.",
        impact="Lower values need more iterations but generalize better.",
        recommended=0.1,
    ),
    "num_leaves": HyperparamDef(
        "num_leaves", "int", 2, 1024, 1, 31,
        description="Maximum number of leaves in one tree.",
        impact="More leaves → more complex model.",
        recommended=31,
    ),
    "subsample": HyperparamDef(
        "subsample", "float", 0.1, 1.0, 0.05, 0.8,
        description="Fraction of data used per iteration.",
        impact="Lower values add randomness, reducing overfitting.",
        recommended=0.8,
    ),
    "reg_alpha": HyperparamDef(
        "reg_alpha", "float", 0.0, 100.0, 0.1, 0.0,
        description="L1 regularization term on weights.",
        impact="Higher values push less-important feature weights to zero.",
        recommended=0.0,
    ),
    "reg_lambda": HyperparamDef(
        "reg_lambda", "float", 0.0, 100.0, 0.1, 0.0,
        description="L2 regularization term on weights.",
        impact="Higher values reduce model complexity.",
        recommended=0.0,
    ),
}

_GBR_PARAMS = {
    "n_estimators": HyperparamDef(
        "n_estimators", "int", 10, MAX_TREES, 10, 100,
        description="Number of boosting stages.",
        impact="More stages improve fit but increase training time.",
        recommended=100,
    ),
    "max_depth": HyperparamDef(
        "max_depth", "int", 1, MAX_DEPTH, 1, 3,
        description="Maximum depth of each tree.",
        impact="Deeper trees capture more interactions.",
        recommended=3,
    ),
    "learning_rate": HyperparamDef(
        "learning_rate", "float", 0.0001, 2.0, 0.01, 0.1,
        description="Shrinks contribution of each tree.",
        impact="Lower values need more trees but generalize better.",
        recommended=0.1,
    ),
    "subsample": HyperparamDef(
        "subsample", "float", 0.1, 1.0, 0.05, 1.0,
        description="Fraction of samples used per tree.",
        impact="Lower values add stochastic gradient boosting.",
        recommended=1.0,
    ),
    "min_samples_split": HyperparamDef(
        "min_samples_split", "int", 2, 20, 1, 2,
        description="Minimum samples to split an internal node.",
        impact="Higher values prevent learning overly specific patterns.",
        recommended=2,
    ),
}

_DT_PARAMS = {
    "max_depth": HyperparamDef(
        "max_depth", "int", 1, MAX_DEPTH, 1, 5,
        description="Maximum tree depth.",
        impact="Deeper trees fit training data better but may overfit.",
        recommended=5,
    ),
    "min_samples_split": HyperparamDef(
        "min_samples_split", "int", 2, 20, 1, 2,
        description="Minimum samples required to split a node.",
        impact="Higher values create a simpler tree.",
        recommended=2,
    ),
    "min_samples_leaf": HyperparamDef(
        "min_samples_leaf", "int", 1, 20, 1, 1,
        description="Minimum samples at a leaf node.",
        impact="Higher values smooth predictions.",
        recommended=1,
    ),
}

_ANN_PARAMS = {
    "hidden_layer_sizes": HyperparamDef(
        "hidden_layer_sizes", "select",
        options=["64", "128", "64,32", "128,64", "128,64,32", "256,128,64"],
        default="128,64",
        description="Number of neurons in each hidden layer (comma-separated).",
        impact="Wider/deeper networks learn more complex patterns but risk overfitting.",
        recommended="128,64",
    ),
    "learning_rate_init": HyperparamDef(
        "learning_rate_init", "float", 0.00001, 1.0, 0.0001, 0.001,
        description="Initial learning rate for the optimizer.",
        impact="Lower values train slower but more precisely.",
        recommended=0.001,
    ),
    "max_iter": HyperparamDef(
        "max_iter", "int", 50, MAX_ANN_ITER, 50, 500,
        description="Maximum number of training epochs.",
        impact="More epochs allow convergence but increase training time.",
        recommended=500,
    ),
    "alpha": HyperparamDef(
        "alpha", "float", 0.00001, 100.0, 0.0001, 0.0001,
        description="L2 regularization term.",
        impact="Higher alpha reduces overfitting.",
        recommended=0.0001,
    ),
    "activation": HyperparamDef(
        "activation", "select", options=["relu", "tanh", "logistic"],
        default="relu",
        description="Activation function for hidden layers.",
        impact="ReLU is fastest; tanh for bounded outputs.",
        recommended="relu",
    ),
}

_STACKING_PARAMS = {
    "final_estimator": HyperparamDef(
        "final_estimator", "select",
        options=["ridge", "linear", "svr"],
        default="ridge",
        description="Meta-learner that combines base model predictions.",
        impact="Ridge is robust for small datasets; SVR for non-linear combinations.",
        recommended="ridge",
    ),
    "cv": HyperparamDef(
        "cv", "int", 2, 10, 1, 5,
        description="Number of cross-validation folds for generating base model predictions.",
        impact="More folds → less bias but slower training.",
        recommended=5,
    ),
    "rf_n_estimators": HyperparamDef(
        "rf_n_estimators", "int", 10, MAX_TREES, 10, 50,
        description="Number of trees in the underlying Random Forest.",
        impact="More trees make the RF base model more robust.",
        recommended=50,
    ),
    "xgb_n_estimators": HyperparamDef(
        "xgb_n_estimators", "int", 10, MAX_TREES, 10, 50,
        description="Number of boosting rounds for the underlying XGBoost.",
        impact="More rounds increase XGBoost power but slow down stacking.",
        recommended=50,
    ),
    "xgb_learning_rate": HyperparamDef(
        "xgb_learning_rate", "float", 0.001, 1.0, 0.01, 0.1,
        description="Learning rate for the underlying XGBoost.",
        impact="Lower values require more estimators to generalize well.",
        recommended=0.1,
    ),
    "svr_C": HyperparamDef(
        "svr_C", "float", 0.01, MAX_SVR_C, 0.1, 1.0,
        description="Regularization (C) for the underlying SVR.",
        impact="Higher values make the base SVR fit closer to training data.",
        recommended=1.0,
    ),
    "final_reg": HyperparamDef(
        "final_reg", "float", 0.01, MAX_SVR_C, 0.1, 1.0,
        description="Regularization strength for the meta-learner (Alpha for Ridge, C for SVR).",
        impact="Higher values increase/decrease regularization depending on the final estimator type.",
        recommended=1.0,
    ),
}

# ---------------------------------------------------------------------------
# Algorithm Registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, AlgorithmMeta] = {
    "xgboost": AlgorithmMeta(
        key="xgboost", display_name="XGBoost",
        description="Gradient boosted trees with regularization. Excellent for tabular data.",
        supports_convergence=True, hyperparameters=_XGBOOST_PARAMS,
    ),
    "random_forest": AlgorithmMeta(
        key="random_forest", display_name="Random Forest",
        description="Ensemble of independent decision trees. Robust and hard to overfit.",
        supports_convergence=False, hyperparameters=_RF_PARAMS,
    ),
    "svr": AlgorithmMeta(
        key="svr", display_name="SVM/SVR",
        description="Support Vector Regression with kernel trick. Good for small datasets.",
        supports_convergence=False, hyperparameters=_SVR_PARAMS,
    ),
    "lightgbm": AlgorithmMeta(
        key="lightgbm", display_name="LightGBM",
        description="Fast gradient boosting with leaf-wise growth. Memory efficient.",
        supports_convergence=True, hyperparameters=_LIGHTGBM_PARAMS,
    ),
    "gradient_boosting": AlgorithmMeta(
        key="gradient_boosting", display_name="Gradient Boosting",
        description="Scikit-learn gradient boosting. Reliable baseline with staged predictions.",
        supports_convergence=True, hyperparameters=_GBR_PARAMS,
    ),
    "decision_tree": AlgorithmMeta(
        key="decision_tree", display_name="Decision Tree",
        description="Single interpretable tree. Fast but prone to overfitting.",
        supports_convergence=False, hyperparameters=_DT_PARAMS,
    ),
    "ann": AlgorithmMeta(
        key="ann", display_name="Neural Network (ANN)",
        description="Multi-layer perceptron regressor. Captures non-linear relationships.",
        supports_convergence=True, hyperparameters=_ANN_PARAMS,
    ),
    "stacking": AlgorithmMeta(
        key="stacking", display_name="Stacking Ensemble",
        description="Combines RF + XGBoost + SVR as base models with a meta-learner.",
        supports_convergence=False, hyperparameters=_STACKING_PARAMS,
    ),
}


def get_algorithm_list() -> list[dict]:
    """Return serializable list of algorithms with hyperparameter metadata."""
    reload_registry_limits()
    result = []
    for key, meta in ALGORITHM_REGISTRY.items():
        params = {}
        for pname, pdef in meta.hyperparameters.items():
            params[pname] = {
                "type": pdef.type,
                "min": pdef.min_val,
                "max": pdef.max_val,
                "step": pdef.step,
                "default": pdef.default,
                "options": pdef.options,
                "description": pdef.description,
                "impact": pdef.impact,
                "recommended": pdef.recommended,
            }
        result.append({
            "key": key,
            "display_name": meta.display_name,
            "description": meta.description,
            "supports_convergence": meta.supports_convergence,
            "hyperparameters": params,
        })
    return result


def get_defaults(algorithm: str) -> dict[str, Any]:
    """Return default hyperparameters for an algorithm."""
    reload_registry_limits()
    meta = ALGORITHM_REGISTRY.get(algorithm)
    if not meta:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return {name: pdef.default for name, pdef in meta.hyperparameters.items()}


def build_model(algorithm: str, hyperparams: dict[str, Any] | None = None):
    """Build a scikit-learn compatible model with the given hyperparameters."""
    n_jobs = get_safe_n_jobs()
    params = get_defaults(algorithm)
    if hyperparams:
        params.update(hyperparams)

    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=n_jobs, random_state=42, verbosity=0,
        )

    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        mf = params["max_features"]
        if mf in ("sqrt", "log2"):
            max_features = mf
        else:
            max_features = float(mf)
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=max_features,
            n_jobs=n_jobs, random_state=42,
        )

    if algorithm == "svr":
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                kernel=str(params["kernel"]),
                gamma=str(params["gamma"]),
            )),
        ])

    if algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        md = int(params["max_depth"])
        return LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=md,
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            subsample=float(params["subsample"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=n_jobs, random_state=42, verbosity=-1,
        )

    if algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            min_samples_split=int(params["min_samples_split"]),
            random_state=42,
        )

    if algorithm == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
        )

    if algorithm == "ann":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        layers_str = str(params["hidden_layer_sizes"])
        hidden = tuple(int(x) for x in layers_str.split(","))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden,
                learning_rate_init=float(params["learning_rate_init"]),
                max_iter=int(params["max_iter"]),
                alpha=float(params["alpha"]),
                activation=str(params["activation"]),
                random_state=42, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20,
            )),
        ])

    if algorithm == "stacking":
        return _build_stacking(params, n_jobs)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _build_stacking(params: dict, n_jobs: int):
    """Build a stacking ensemble with RF + XGBoost + SVR base estimators."""
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    rf_n = int(params.get("rf_n_estimators", 50))
    xgb_n = int(params.get("xgb_n_estimators", 50))
    xgb_lr = float(params.get("xgb_learning_rate", 0.1))
    svr_c_base = float(params.get("svr_C", 1.0))
    final_reg = float(params.get("final_reg", 1.0))

    base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=rf_n, max_depth=8, n_jobs=n_jobs, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=xgb_n, learning_rate=xgb_lr, max_depth=4, n_jobs=n_jobs, random_state=42, verbosity=0)),
        ("svr", Pipeline([("sc", StandardScaler()), ("svr", SVR(C=svr_c_base, kernel="rbf"))])),
    ]

    fe = str(params.get("final_estimator", "ridge"))
    if fe == "ridge":
        from sklearn.linear_model import Ridge
        final = Ridge(alpha=final_reg)
    elif fe == "svr":
        final = Pipeline([("sc", StandardScaler()), ("svr", SVR(C=final_reg))])
    else:
        from sklearn.linear_model import LinearRegression
        final = LinearRegression()

    return StackingRegressor(
        estimators=base_estimators, final_estimator=final,
        cv=int(params.get("cv", 5)), n_jobs=n_jobs,
    )
