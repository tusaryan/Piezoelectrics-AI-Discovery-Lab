"""
NSGA-II Optimizer — Multi-objective optimization using pymoo.

Uses trained ML models as surrogate fitness functions to find
Pareto-optimal compositions balancing d33, tc, and hardness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from piezo_ml.models.platform_utils import get_safe_n_jobs
from piezo_ml.registry import ELEMENT_REGISTRY

# pymoo imports — guarded for environments without pymoo installed
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as _PymooProblem
    from pymoo.optimize import minimize as _pymoo_minimize
    from pymoo.termination import get_termination as _pymoo_get_termination
    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False
    _PymooProblem = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Elements eligible for optimization search space (non-O, non-rare-earth heavy)
SEARCH_ELEMENTS = [
    sym for sym, props in ELEMENT_REGISTRY.items()
    if sym != "O" and props.get("perovskite_site") in ("A", "B", "dopant")
]

# Use-case presets: target ranges for different applications
USE_CASE_PRESETS = {
    "flexible_wearables": {
        "label": "🔋 Flexible Wearables",
        "description": "High d33 + low hardness + moderate tc",
        "objectives": {
            "d33": {"direction": "maximize", "min": 200, "max": 700, "weight": 1.0},
            "tc": {"direction": "maximize", "min": 150, "max": 350, "weight": 0.5},
            "vickers_hardness": {"direction": "minimize", "min": 50, "max": 300, "weight": 0.7},
        },
    },
    "industrial_actuators": {
        "label": "⚡ Industrial Actuators",
        "description": "Moderate d33 + high hardness + high tc",
        "objectives": {
            "d33": {"direction": "maximize", "min": 100, "max": 500, "weight": 0.7},
            "tc": {"direction": "maximize", "min": 300, "max": 500, "weight": 1.0},
            "vickers_hardness": {"direction": "maximize", "min": 300, "max": 1200, "weight": 0.8},
        },
    },
    "ultrasonic_transducers": {
        "label": "🔊 Ultrasonic Transducers",
        "description": "Moderate d33 + very high hardness + high tc",
        "objectives": {
            "d33": {"direction": "maximize", "min": 80, "max": 400, "weight": 0.6},
            "tc": {"direction": "maximize", "min": 350, "max": 500, "weight": 0.9},
            "vickers_hardness": {"direction": "maximize", "min": 500, "max": 1200, "weight": 1.0},
        },
    },
    "custom": {
        "label": "🎯 Custom",
        "description": "User-defined property ranges",
        "objectives": {
            "d33": {"direction": "maximize", "min": 50, "max": 700, "weight": 1.0},
            "tc": {"direction": "maximize", "min": 100, "max": 500, "weight": 1.0},
            "vickers_hardness": {"direction": "maximize", "min": 50, "max": 1200, "weight": 1.0},
        },
    },
}


@dataclass
class OptimizationConfig:
    """Configuration for NSGA-II optimization run."""
    model_ids: dict[str, str]  # target -> model_id
    objectives: dict[str, dict[str, Any]]  # target -> {direction, min, max, weight}
    preset: str = "custom"
    pop_size: int = 100
    n_generations: int = 50
    seed: int = 42
    search_elements: list[str] | None = None


@dataclass
class ParetoSolution:
    """A single Pareto-optimal solution."""
    composition: dict[str, float]  # element -> fraction
    formula_approx: str
    predicted: dict[str, float]  # target -> predicted value
    use_case_tag: str = ""
    use_case_color: str = ""
    rank: int = 0
    crowding_distance: float = 0.0


@dataclass
class OptimizationResult:
    """Full result of NSGA-II optimization."""
    solutions: list[ParetoSolution]
    convergence: list[dict[str, float]]  # [{generation, hv_indicator, ...}]
    n_generations_run: int = 0
    n_evaluations: int = 0
    duration_seconds: float = 0.0
    targets_optimized: list[str] = field(default_factory=list)
    preset_used: str = "custom"
    error: str | None = None


def _composition_to_formula(comp: dict[str, float], threshold: float = 0.01) -> str:
    """Convert element fractions to approximate formula string."""
    parts = []
    for elem, frac in sorted(comp.items(), key=lambda x: -x[1]):
        if frac < threshold:
            continue
        if abs(frac - 1.0) < 0.01:
            parts.append(elem)
        elif abs(frac - round(frac, 0)) < 0.01:
            parts.append(f"{elem}{int(round(frac))}")
        else:
            parts.append(f"{elem}{frac:.2f}")
    return "".join(parts) + "O3" if parts else "unknown"


class NSGA2Optimizer:
    """Multi-objective optimizer using NSGA-II via pymoo."""

    def __init__(self) -> None:
        self._cancel_flag = False

    def cancel(self) -> None:
        """Signal optimization to stop."""
        self._cancel_flag = True

    def optimize(
        self,
        config: OptimizationConfig,
        models: dict[str, Any],
        feature_columns: dict[str, list[str]],
    ) -> OptimizationResult:
        """Run NSGA-II optimization.

        Args:
            config: Optimization configuration
            models: dict of target -> loaded sklearn model
            feature_columns: dict of target -> list of feature column names
        """
        self._cancel_flag = False
        start = time.time()

        if not _PYMOO_AVAILABLE:
            return OptimizationResult(
                solutions=[],
                convergence=[],
                error="pymoo not installed. Run: pip install pymoo",
            )

        # Determine search elements
        search_elems = config.search_elements or SEARCH_ELEMENTS[:15]
        n_vars = len(search_elems)
        targets = list(config.objectives.keys())
        active_targets = [t for t in targets if t in models]

        if not active_targets:
            return OptimizationResult(
                solutions=[], convergence=[],
                error="No trained models available for the selected targets",
            )

        convergence_log: list[dict[str, float]] = []

        # Define the optimization problem
        problem = _PiezoOptProblem(
            search_elems=search_elems,
            targets=active_targets,
            objectives=config.objectives,
            models=models,
            feature_columns=feature_columns,
            convergence_log=convergence_log,
            cancel_flag=lambda: self._cancel_flag,
        )

        algorithm = NSGA2(pop_size=config.pop_size)
        termination = _pymoo_get_termination("n_gen", config.n_generations)

        try:
            res = _pymoo_minimize(
                problem,
                algorithm,
                termination,
                seed=config.seed,
                verbose=False,
                save_history=False,
            )
        except _CancelledError:
            return OptimizationResult(
                solutions=[], convergence=convergence_log,
                error="Optimization cancelled by user",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            logger.error(f"NSGA-II optimization failed: {e}", exc_info=True)
            return OptimizationResult(
                solutions=[], convergence=convergence_log,
                error=f"Optimization failed: {str(e)}",
                duration_seconds=time.time() - start,
            )

        # Extract Pareto front
        solutions = _extract_solutions(
            res, search_elems, active_targets, config.objectives,
        )

        return OptimizationResult(
            solutions=solutions,
            convergence=convergence_log,
            n_generations_run=res.algorithm.n_gen if res.algorithm else 0,
            n_evaluations=res.algorithm.evaluator.n_eval if res.algorithm and res.algorithm.evaluator else 0,
            duration_seconds=round(time.time() - start, 2),
            targets_optimized=active_targets,
            preset_used=config.preset,
        )


class _CancelledError(Exception):
    """Raised when optimization is cancelled."""
    pass


class _PiezoOptProblem(_PymooProblem):
    """pymoo Problem: element fractions → predicted properties."""

    def __init__(
        self,
        search_elems: list[str],
        targets: list[str],
        objectives: dict[str, dict],
        models: dict[str, Any],
        feature_columns: dict[str, list[str]],
        convergence_log: list[dict],
        cancel_flag,
    ) -> None:
        self.search_elems = search_elems
        self.targets = targets
        self.objectives = objectives
        self.models = models
        self.feature_columns = feature_columns
        self.convergence_log = convergence_log
        self.cancel_flag = cancel_flag
        self._gen_count = 0

        # Feature engineering helpers
        from piezo_ml.features import FeatureEngineer
        self.engineer = FeatureEngineer()

        # Variables: element fractions (sum ≈ 1)
        # Bounds: [0, 1] for each element
        n_var = len(search_elems)
        n_obj = len(targets)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate fitness for a population of compositions."""
        if self.cancel_flag():
            raise _CancelledError()

        self._gen_count += 1
        n_pop = X.shape[0]
        F = np.full((n_pop, len(self.targets)), 1e6)

        for i in range(n_pop):
            # Normalize fractions to sum to 1
            raw = X[i]
            total = raw.sum()
            if total <= 0:
                continue
            fractions = raw / total

            # Build composition dict
            comp = {
                self.search_elems[j]: float(fractions[j])
                for j in range(len(self.search_elems))
                if fractions[j] > 0.005
            }

            # Predict each target
            for t_idx, target in enumerate(self.targets):
                model = self.models.get(target)
                feat_cols = self.feature_columns.get(target, [])
                if model is None:
                    continue

                try:
                    fv = self._build_features(comp, feat_cols)
                    pred = model.predict(fv)[0]
                except Exception:
                    pred = 0.0

                # Convert to minimization (pymoo minimizes)
                obj = self.objectives.get(target, {})
                direction = obj.get("direction", "maximize")
                weight = obj.get("weight", 1.0)

                if direction == "maximize":
                    F[i, t_idx] = -pred * weight
                else:
                    F[i, t_idx] = pred * weight

        out["F"] = F

        # Log convergence
        if self._gen_count % 5 == 0 or self._gen_count == 1:
            avg_objectives = {}
            for t_idx, target in enumerate(self.targets):
                avg_objectives[target] = float(np.mean(np.abs(F[:, t_idx])))
            self.convergence_log.append({
                "generation": self._gen_count,
                **avg_objectives,
            })

    def _build_features(
        self, comp: dict[str, float], feat_cols: list[str],
    ) -> np.ndarray:
        """Build feature vector from composition for a given model."""
        features: dict[str, float] = {}

        # Element fractions
        for elem, frac in comp.items():
            features[f"frac_{elem}"] = frac

        # Weighted physics properties
        from piezo_ml.registry import PROPERTY_KEYS
        skip_props = {"symbol", "oxidation_states", "block", "perovskite_site", "is_rare_earth"}
        for prop in PROPERTY_KEYS:
            if prop in skip_props:
                continue
            vals = []
            for elem, frac in comp.items():
                raw = ELEMENT_REGISTRY.get(elem, {}).get(prop)
                try:
                    v = float(raw)
                    if not (np.isnan(v) or np.isinf(v)):
                        vals.append((v, frac))
                except (TypeError, ValueError):
                    continue
            if vals:
                total_w = sum(w for _, w in vals)
                mean = sum(v * w for v, w in vals) / total_w if total_w > 0 else 0
                var = sum(w * (v - mean) ** 2 for v, w in vals) / total_w if total_w > 0 else 0
            else:
                mean, var = 0.0, 0.0
            features[f"{prop}_weighted_mean"] = mean
            features[f"{prop}_weighted_var"] = var

        # Structural factors (tolerance + octahedral)
        import math
        o_radius = ELEMENT_REGISTRY.get("O", {}).get("ionic_radius_pm")
        try:
            o_radius = float(o_radius) if o_radius else 0.0
        except (TypeError, ValueError):
            o_radius = 0.0

        a_r, b_r = [], []
        for elem, frac in comp.items():
            props = ELEMENT_REGISTRY.get(elem, {})
            site = props.get("perovskite_site")
            r = props.get("ionic_radius_pm")
            try:
                r = float(r) if r else None
            except (TypeError, ValueError):
                r = None
            if r is None:
                continue
            if site == "A":
                a_r.append((r, frac))
            elif site == "B":
                b_r.append((r, frac))

        r_a = sum(v * w for v, w in a_r) / sum(w for _, w in a_r) if a_r else 0
        r_b = sum(v * w for v, w in b_r) / sum(w for _, w in b_r) if b_r else 0

        if o_radius > 0 and r_a > 0 and r_b > 0:
            features["tolerance_factor"] = (r_a + o_radius) / (math.sqrt(2) * (r_b + o_radius))
            features["octahedral_factor"] = r_b / o_radius
        else:
            features["tolerance_factor"] = 0.0
            features["octahedral_factor"] = 0.0

        # Composite features (all zero for optimization — we optimize ceramics)
        composite_keys = [
            "filler_wt_pct", "particle_size_nm",
            "matrix_type_encoded", "particle_morphology_encoded",
            "surface_treatment_encoded", "fabrication_method_encoded",
            "sintering_temp_c", "relative_density_pct",
        ]
        for k in composite_keys:
            features[k] = 0.0

        # Align to model feature columns
        row = {col: features.get(col, 0.0) for col in feat_cols}
        return np.array([list(row.values())])


def _extract_solutions(
    res, search_elems: list[str], targets: list[str],
    objectives: dict[str, dict],
) -> list[ParetoSolution]:
    """Extract and rank Pareto-optimal solutions from pymoo result."""
    from piezo_ml.optimization.pareto_utils import tag_use_case

    if res.X is None or res.F is None:
        return []

    X = res.X
    F = res.F
    solutions = []

    for i in range(len(X)):
        raw = X[i]
        total = raw.sum()
        if total <= 0:
            continue
        fracs = raw / total

        comp = {}
        for j, elem in enumerate(search_elems):
            if fracs[j] > 0.005:
                comp[elem] = round(float(fracs[j]), 4)

        predicted = {}
        for t_idx, target in enumerate(targets):
            obj = objectives.get(target, {})
            direction = obj.get("direction", "maximize")
            weight = obj.get("weight", 1.0)
            val = float(F[i, t_idx])
            if direction == "maximize":
                predicted[target] = round(-val / weight, 2)
            else:
                predicted[target] = round(val / weight, 2)

        tag, color = tag_use_case(predicted)

        solutions.append(ParetoSolution(
            composition=comp,
            formula_approx=_composition_to_formula(comp),
            predicted=predicted,
            use_case_tag=tag,
            use_case_color=color,
            rank=i + 1,
            crowding_distance=0.0,
        ))

    # Sort by first objective (descending if maximize)
    first_target = targets[0]
    first_dir = objectives.get(first_target, {}).get("direction", "maximize")
    solutions.sort(
        key=lambda s: s.predicted.get(first_target, 0),
        reverse=(first_dir == "maximize"),
    )
    for idx, s in enumerate(solutions):
        s.rank = idx + 1

    return solutions
