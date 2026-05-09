"""
OptunaTuner — automatic hyperparameter optimization per algorithm.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from piezo_ml.models.algorithm_registry import ALGORITHM_REGISTRY, build_model
from piezo_ml.models.platform_utils import get_safe_n_jobs
from piezo_ml.models.trainer import TrainingCancelledError


class OptunaTuner:
    """Find optimal hyperparameters via Optuna Bayesian optimization."""

    def __init__(
        self,
        algorithm: str,
        target: str,
        n_trials: int = 30,
        cancel_event: threading.Event | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.target = target
        self.n_trials = n_trials
        self.cancel_event = cancel_event
        self.log_callback = log_callback

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, Any]:
        """Run Optuna optimization and return best hyperparameters."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            if self.cancel_event and self.cancel_event.is_set():
                raise TrainingCancelledError("Tuning cancelled by user")
            params = self._suggest_params(trial)
            model = build_model(self.algorithm, params)
            n_jobs = get_safe_n_jobs()
            cv = min(5, len(X_train))
            if cv < 2:
                cv = 2
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="r2", n_jobs=n_jobs)
            mean_r2 = float(scores.mean())
            self._log("info",
                      f"Optuna trial {trial.number}: R²={mean_r2:.4f} params={params}")
            return mean_r2

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        best = study.best_params
        self._log("success",
                  f"Optuna best for {self.algorithm}/{self.target}: "
                  f"R²={study.best_value:.4f} → {best}")
        return best

    # ----- param suggestion -----

    def _suggest_params(self, trial) -> dict[str, Any]:
        meta = ALGORITHM_REGISTRY.get(self.algorithm)
        if not meta:
            return {}
        params: dict[str, Any] = {}
        for name, pdef in meta.hyperparameters.items():
            if pdef.type == "int":
                params[name] = trial.suggest_int(
                    name, int(pdef.min_val or 1), int(pdef.max_val or 100),
                    step=max(1, int(pdef.step or 1)),
                )
            elif pdef.type == "float":
                params[name] = trial.suggest_float(
                    name, float(pdef.min_val or 0.0), float(pdef.max_val or 1.0),
                    step=float(pdef.step) if pdef.step else None,
                )
            elif pdef.type == "select" and pdef.options:
                params[name] = trial.suggest_categorical(name, pdef.options)
        return params

    def _log(self, level: str, message: str) -> None:
        if self.log_callback:
            self.log_callback(level, message)
