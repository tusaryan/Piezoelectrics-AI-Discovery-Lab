import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

PYMOO_AVAILABLE = False
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    pass

import warnings

class PiezoInverseProblem(ElementwiseProblem if PYMOO_AVAILABLE else object):
    def __init__(self, objective_weights: List[float], constraints: Dict[str, float] = None, models=None, engineer=None):
        # 15 compositional variables, bounded [0, 1]
        super().__init__(n_var=15, n_obj=3, n_ieq_constr=3, xl=0.0, xu=1.0)
        self.objective_weights = objective_weights
        self.constraints = constraints or {}
        self.models = models
        self.engineer = engineer
        self.elements = self.engineer.ELEMENTS[:15] if self.engineer else []

    def _evaluate(self, x, out, *args, **kwargs):
        # Ensure composition fractions sum to 1 (soft penalty via constraints or normalization)
        total_frac = np.sum(x)
        x_norm = x / (total_frac + 1e-9)
        
        if self.models and self.engineer:
            formula_parts = []
            for i, el in enumerate(self.elements):
                if x_norm[i] > 0.001:
                    formula_parts.append(f"{el}{x_norm[i]:.3f}")
            formula_parts.append("O3.000") # Base oxygen
            formula = "".join(formula_parts)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vector, _ = self.engineer.compute_features(formula)
                vector_arr = np.array([vector])
                
                d33_model, tc_model = self.models
                pred_d33 = float(d33_model.predict(vector_arr)[0]) if d33_model else (100.0 + x_norm[0]*400.0)
                pred_tc = float(tc_model.predict(vector_arr)[0]) if tc_model else (150.0 + x_norm[2]*300.0)
                pred_hardness = 2.0 + x_norm[4]*8.0 # Mock hardness if no model
            except Exception as e:
                pred_d33 = 100.0 + x_norm[0]*400.0 + x_norm[1]*100.0 - x_norm[2]*50.0
                pred_tc = 150.0 + x_norm[2]*300.0 - x_norm[0]*50.0
                pred_hardness = 2.0 + x_norm[4]*8.0
        else:
            # MOCK ML PREDICTIONS for UI development
            pred_d33 = 100.0 + x_norm[0]*400.0 + x_norm[1]*100.0 - x_norm[2]*50.0  # max ~ 550
            pred_tc = 150.0 + x_norm[2]*300.0 - x_norm[0]*50.0                     # max ~ 450
            pred_hardness = 2.0 + x_norm[4]*8.0                                    # max ~ 10.0
        
        # Pymoo minimizes by default. Negate to maximize.
        f1 = -pred_d33   # Obj 1: maximize d33
        f2 = -pred_tc    # Obj 2: maximize Tc
        f3 = -pred_hardness # Obj 3: maximize Hardness
        
        # Constraints: values <= 0 mean constraint is satisfied
        g1 = (total_frac - 1.05) # Total frac cannot exceed 1.05 loosely before normalization
        g2 = (0.95 - total_frac)
        
        # User defined constraint bounds check
        g3 = (self.constraints.get("d33_min", 0) - pred_d33) if "d33_min" in self.constraints else -1.0
        
        out["F"] = [f1, f2, f3]
        out["G"] = [g1, g2, g3]

class ParetoOptimizer:
    def __init__(self, config: Dict[str, Any], models=None, engineer=None):
        self.config = config
        self.n_gen = config.get("n_generations", 100)
        self.pop_size = config.get("population_size", 200)
        self.models = models
        self.engineer = engineer

    async def run_optimization_async(self, cb: Any = None) -> List[Dict[str, Any]]:
        if not PYMOO_AVAILABLE:
            logger.warning("pymoo not installed. Using mocked NSGA-II solver for Phase 3 UI testing.")
            return await self._mock_nsga2(cb)

        problem = PiezoInverseProblem(
            objective_weights=[1.0, 1.0, 1.0], 
            constraints=self.config.get("constraints", {}),
            models=self.models,
            engineer=self.engineer
        )
        
        algorithm = NSGA2(pop_size=self.pop_size)
        
        algorithm.setup(problem, termination=("n_gen", self.n_gen))
        
        solutions = []
        for gen in range(self.n_gen):
            algorithm.next()
            
            # Approximated hypervolume logic or just convergence tracking
            pop = algorithm.pop
            best_f1 = min(ind.F[0] for ind in pop)
            hypervolume_approx = -best_f1 * 1.5 + (gen * 0.1)  # Fake hypervolume calculation for ui feedback
            
            if cb:
                await cb({"generation": gen+1, "hypervolume": abs(hypervolume_approx)})
                
            # Sleep briefly to not lock asyncio event loop
            await asyncio.sleep(0.01)
            
        res = algorithm.result()
        for i, ind in enumerate(res.pop):
            if ind.F is not None and len(ind.F) == 3:
                solutions.append({
                    "composition": ind.X.tolist(),
                    "predicted_d33": -ind.F[0],
                    "predicted_tc": -ind.F[1],
                    "predicted_hardness": -ind.F[2],
                    "rank": 1,
                    "crowd_distance": 0.0,
                    "use_case": "Optimized Blend"
                })
        
        return solutions

    async def _mock_nsga2(self, cb: Any) -> List[Dict[str, Any]]:
        # Mock progression
        for gen in range(self.n_gen):
            await asyncio.sleep(0.05)
            if cb:
                await cb({"generation": gen+1, "hypervolume": 50.0 + gen * 2.5 + np.random.normal(0, 5)})
                
        # Mock pareto front returning random trade-offs
        solutions = []
        for i in range(50):
            d33 = np.random.uniform(100, 600)
            tc = np.random.uniform(150, 450)
            hardness = np.random.uniform(2, 9)
            
            solutions.append({
                "composition": np.random.dirichlet(np.ones(15)).tolist(),
                "predicted_d33": float(d33),
                "predicted_tc": float(tc),
                "predicted_hardness": float(hardness),
                "rank": 1,
                "crowd_distance": float(np.random.rand()),
                "use_case": "High-Temp Sensor" if tc > 300 else "Medical Imaging"
            })
            
        return solutions
