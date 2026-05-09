"""
ModelTrainer — trains a single sklearn-compatible model with convergence
tracking, cancellation support, and progress callbacks.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from piezo_ml.models.algorithm_registry import build_model


class TrainingCancelledError(Exception):
    """Raised when training is cancelled by the user."""


@dataclass
class TrainingResult:
    """Result of a single model training run."""
    target: str
    algorithm: str
    model: Any  # fitted sklearn-compatible model
    r2: float
    rmse: float
    hyperparameters: dict[str, Any]
    convergence_data: list[dict[str, float]]  # [{iteration, metric}]
    training_duration_s: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0


class ModelTrainer:
    """Trains one model for one target."""

    def __init__(
        self,
        algorithm: str,
        target: str,
        hyperparameters: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.target = target
        self.hyperparameters = hyperparameters or {}
        self.cancel_event = cancel_event
        self.log_callback = log_callback

    # ----- public -----

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        self._check_cancelled()
        self._log("info", f"Training {self.algorithm} for {self.target}...")
        start = time.time()

        model = build_model(self.algorithm, self.hyperparameters)

        convergence = self._fit_with_convergence(model, X_train, y_train, X_test, y_test)

        self._check_cancelled()
        y_pred = self._predict(model, X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
        duration = time.time() - start

        importances = self._extract_importances(model, feature_names)

        self._log("success",
                  f"{self.algorithm} for {self.target}: R²={r2:.4f}, RMSE={rmse:.2f} ({duration:.1f}s)")

        return TrainingResult(
            target=self.target, algorithm=self.algorithm, model=model,
            r2=r2, rmse=rmse, hyperparameters=self.hyperparameters,
            convergence_data=convergence, training_duration_s=duration,
            feature_importances=importances,
            n_train=len(X_train), n_test=len(X_test),
        )

    # ----- convergence tracking -----

    def _fit_with_convergence(
        self, model: Any,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> list[dict[str, float]]:
        convergence: list[dict[str, float]] = []

        if self.algorithm == "xgboost":
            convergence = self._fit_xgboost(model, X_train, y_train, X_test, y_test)
        elif self.algorithm == "lightgbm":
            convergence = self._fit_lightgbm(model, X_train, y_train, X_test, y_test)
        elif self.algorithm == "gradient_boosting":
            convergence = self._fit_gbr(model, X_train, y_train, X_test, y_test)
        elif self.algorithm == "ann":
            convergence = self._fit_ann(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)
        return convergence

    def _fit_xgboost(self, model, X_train, y_train, X_test, y_test):
        eval_set = [(X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        results = model.evals_result()
        metric_key = list(results.get("validation_0", {}).keys())
        if metric_key:
            vals = results["validation_0"][metric_key[0]]
            return [{"iteration": i, "metric": float(v)} for i, v in enumerate(vals)]
        return []

    def _fit_lightgbm(self, model, X_train, y_train, X_test, y_test):
        callbacks = []
        convergence: list[dict[str, float]] = []

        def _record(env):
            if env.evaluation_result_list:
                _, _, val, _ = env.evaluation_result_list[0]
                convergence.append({"iteration": env.iteration, "metric": float(val)})

        callbacks.append(_record)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        return convergence

    def _fit_gbr(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        convergence = []
        if hasattr(model, "staged_predict"):
            for i, y_pred in enumerate(model.staged_predict(X_test)):
                r2 = float(r2_score(y_test, y_pred))
                convergence.append({"iteration": i, "metric": r2})
        return convergence

    def _fit_ann(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        convergence = []
        # MLPRegressor stores loss_curve_ on the inner estimator
        mlp = model.named_steps.get("mlp", model) if hasattr(model, "named_steps") else model
        if hasattr(mlp, "loss_curve_"):
            for i, loss in enumerate(mlp.loss_curve_):
                convergence.append({"iteration": i, "metric": float(loss)})
        return convergence

    # ----- helpers -----

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def _extract_importances(self, model: Any, names: list[str] | None) -> dict[str, float]:
        estimator = model
        if hasattr(model, "named_steps"):
            for step_name in reversed(list(model.named_steps)):
                step = model.named_steps[step_name]
                if hasattr(step, "feature_importances_"):
                    estimator = step
                    break
        if not hasattr(estimator, "feature_importances_"):
            return {}
        imp = estimator.feature_importances_
        if names and len(names) == len(imp):
            return {n: float(v) for n, v in zip(names, imp)}
        return {f"feature_{i}": float(v) for i, v in enumerate(imp)}

    def _check_cancelled(self) -> None:
        if self.cancel_event and self.cancel_event.is_set():
            raise TrainingCancelledError("Training cancelled by user")

    def _log(self, level: str, message: str) -> None:
        if self.log_callback:
            self.log_callback(level, message)
