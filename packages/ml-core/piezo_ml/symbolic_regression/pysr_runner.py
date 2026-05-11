"""
PySR Runner — discover interpretable equations via symbolic regression.

PySR requires Julia backend — this module handles graceful fallback
when Julia/PySR is not installed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EquationResult:
    """A single discovered equation from PySR."""
    equation_str: str          # e.g., "x0 * 2.5 + x1^2"
    latex: str                 # LaTeX form for KaTeX rendering
    complexity: int            # number of operations
    loss: float                # training loss
    r2: float                  # R² on training data
    feature_mapping: dict[str, str]  # x0 → "tolerance_factor", etc.
    readable: str              # equation with real feature names


@dataclass
class SymbolicRegressionResult:
    """Complete PySR result with Pareto front of equations."""
    target: str
    equations: list[EquationResult]
    best_equation: EquationResult | None
    pareto_front: list[dict[str, float]]  # [{complexity, loss, r2}]
    n_samples: int
    n_features: int
    available: bool = True
    error: str | None = None


def _check_pysr_available() -> bool:
    """Check if PySR and Julia are available."""
    try:
        import pysr  # noqa: F401
        return True
    except ImportError:
        return False


class PySRRunner:
    """Run symbolic regression using PySR."""

    def __init__(self) -> None:
        self._available = _check_pysr_available()

    @property
    def available(self) -> bool:
        return self._available

    def run(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        target: str = "d33",
        max_complexity: int = 20,
        n_iterations: int = 40,
        populations: int = 15,
        timeout_seconds: int = 120,
    ) -> SymbolicRegressionResult:
        """Run PySR symbolic regression."""
        if not self._available:
            return SymbolicRegressionResult(
                target=target,
                equations=[],
                best_equation=None,
                pareto_front=[],
                n_samples=len(X),
                n_features=len(X.columns),
                available=False,
                error="PySR is not installed. Install with: pip install 'piezo-ml[symbolic]' "
                      "and ensure Julia is available.",
            )

        try:
            return self._run_pysr(
                X, y, target, max_complexity, n_iterations,
                populations, timeout_seconds,
            )
        except Exception as e:
            logger.error(f"PySR failed: {e}")
            return SymbolicRegressionResult(
                target=target,
                equations=[],
                best_equation=None,
                pareto_front=[],
                n_samples=len(X),
                n_features=len(X.columns),
                available=True,
                error=f"Symbolic regression failed: {str(e)}",
            )

    def _run_pysr(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        target: str,
        max_complexity: int,
        n_iterations: int,
        populations: int,
        timeout_seconds: int,
    ) -> SymbolicRegressionResult:
        """Internal PySR execution."""
        from pysr import PySRRegressor

        feature_names = list(X.columns)
        feature_mapping = {f"x{i}": name for i, name in enumerate(feature_names)}

        # Limit features for tractability (top 10 by variance)
        if len(feature_names) > 10:
            variances = X.var().sort_values(ascending=False)
            top_features = variances.head(10).index.tolist()
            X = X[top_features]
            feature_names = top_features
            feature_mapping = {f"x{i}": name for i, name in enumerate(feature_names)}

        model = PySRRegressor(
            niterations=n_iterations,
            populations=populations,
            maxsize=max_complexity,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "abs"],
            timeout_in_seconds=timeout_seconds,
            temp_equation_file=True,
            verbosity=0,
            progress=False,
            deterministic=True,
            random_state=42,
            procs=1,       # single process for macOS safety
        )

        y_arr = np.asarray(y, dtype=float)
        model.fit(X.values, y_arr, variable_names=feature_names)

        # Extract equations from Pareto front
        equations: list[EquationResult] = []
        pareto_front: list[dict[str, float]] = []

        if hasattr(model, "equations_") and model.equations_ is not None:
            eq_df = model.equations_
            for _, row in eq_df.iterrows():
                eq_str = str(row.get("equation", ""))
                complexity = int(row.get("complexity", 0))
                loss = float(row.get("loss", 0))

                # Compute R² for this equation
                try:
                    y_pred = model.predict(X.values, index=int(row.name))
                    ss_res = np.sum((y_arr - y_pred) ** 2)
                    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                except Exception:
                    r2 = 0.0

                latex = _equation_to_latex(eq_str, feature_names)
                readable = _substitute_features(eq_str, feature_mapping)

                eq_result = EquationResult(
                    equation_str=eq_str,
                    latex=latex,
                    complexity=complexity,
                    loss=round(loss, 6),
                    r2=round(r2, 4),
                    feature_mapping=feature_mapping,
                    readable=readable,
                )
                equations.append(eq_result)
                pareto_front.append({
                    "complexity": complexity,
                    "loss": round(loss, 6),
                    "r2": round(r2, 4),
                })

        best = max(equations, key=lambda e: e.r2) if equations else None

        return SymbolicRegressionResult(
            target=target,
            equations=equations,
            best_equation=best,
            pareto_front=pareto_front,
            n_samples=len(X),
            n_features=len(feature_names),
        )


def _equation_to_latex(eq_str: str, feature_names: list[str]) -> str:
    """Convert PySR equation string to LaTeX."""
    latex = eq_str

    # Replace feature names with LaTeX-safe versions
    for name in sorted(feature_names, key=len, reverse=True):
        safe = name.replace("_", r"\_")
        latex = latex.replace(name, rf"\text{{{safe}}}")

    # Common transformations
    latex = latex.replace("square(", r"\text{square}(")
    latex = latex.replace("sqrt(", r"\sqrt{")
    # Close sqrt braces
    if r"\sqrt{" in latex:
        latex = _balance_sqrt(latex)

    latex = latex.replace("**", "^")
    latex = latex.replace("*", r" \cdot ")
    latex = latex.replace("abs(", r"|")

    return latex


def _balance_sqrt(latex: str) -> str:
    """Balance sqrt braces in LaTeX string."""
    result = []
    i = 0
    while i < len(latex):
        if latex[i:].startswith(r"\sqrt{"):
            result.append(r"\sqrt{")
            i += 6
            depth = 1
            while i < len(latex) and depth > 0:
                if latex[i] == "{":
                    depth += 1
                elif latex[i] == ")":
                    if depth == 1:
                        result.append("}")
                        depth -= 1
                        i += 1
                        continue
                elif latex[i] == "}":
                    depth -= 1
                result.append(latex[i])
                i += 1
        else:
            result.append(latex[i])
            i += 1
    return "".join(result)


def _substitute_features(eq_str: str, mapping: dict[str, str]) -> str:
    """Replace x0, x1, etc. with real feature names."""
    readable = eq_str
    for var, name in sorted(mapping.items(), key=lambda x: -len(x[0])):
        readable = readable.replace(var, name)
    return readable
