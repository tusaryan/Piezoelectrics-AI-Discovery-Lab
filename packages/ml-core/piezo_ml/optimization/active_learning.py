import numpy as np
import logging
import asyncio
from typing import Dict, List, Any, Callable
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

logger = logging.getLogger(__name__)

class ActiveLearningSimulator:
    """
    Simulates the Active Learning loop:
    - Generates a synthetic pool of unseen candidates
    - Uses a RandomForest surrogate to calculate Uncertainty (Std) and Means
    - Iteratively selects candidates via an acquisition function (UCB, EI, Random)
    - Compares strategy vs baseline random sampling
    """
    
    def __init__(self, data: Any = None, total_budget: int = 50, strategy: str = "UCB", acquisition_fn: str = "UCB", oracle_model=None):
        self.data = data
        self.total_budget = total_budget
        self.strategy = strategy
        self.acquisition_fn = acquisition_fn
        self.oracle_model = oracle_model
        
    async def simulate_async(self, cb: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        strategy_curve = []
        baseline_curve = []
        
        # 1. Generate Synthetic Candidate Pool (500 candidates, 35 features)
        np.random.seed(42)
        pool_size = 500
        X_pool = np.random.rand(pool_size, 35)
        
        # 2. Get "True" labels from Oracle (or mock if no oracle)
        if self.oracle_model:
            true_y_pool = self.oracle_model.predict(X_pool)
        else:
            # Mock complex non-linear underlying function
            true_y_pool = 100 + X_pool[:,0]*400 + X_pool[:,1]*300 - X_pool[:,2]*200 + np.sin(X_pool[:,3]*10)*100
        
        true_max = float(np.max(true_y_pool))
        
        # 3. Initialize Training Sets (Start with 5 random samples)
        initial_indices = np.random.choice(pool_size, 5, replace=False)
        
        X_train_strat = X_pool[initial_indices].tolist()
        y_train_strat = true_y_pool[initial_indices].tolist()
        
        X_train_base = X_pool[initial_indices].tolist()
        y_train_base = true_y_pool[initial_indices].tolist()
        
        # Remaining pools
        strat_pool_mask = np.ones(pool_size, dtype=bool)
        strat_pool_mask[initial_indices] = False
        
        base_pool_mask = np.ones(pool_size, dtype=bool)
        base_pool_mask[initial_indices] = False
        
        current_strategy_best = float(np.max(y_train_strat))
        current_baseline_best = float(np.max(y_train_base))
        
        surrogate = RandomForestRegressor(n_estimators=50, random_state=42)
        
        found_max_d33_iter = None
        
        for i in range(1, self.total_budget + 1):
            if not np.any(strat_pool_mask) or not np.any(base_pool_mask):
                break
                
            # BASELINE (Random Selection)
            base_available_idx = np.where(base_pool_mask)[0]
            chosen_base = np.random.choice(base_available_idx)
            
            X_train_base.append(X_pool[chosen_base])
            y_train_base.append(true_y_pool[chosen_base])
            base_pool_mask[chosen_base] = False
            
            if true_y_pool[chosen_base] > current_baseline_best:
                current_baseline_best = float(true_y_pool[chosen_base])
                
            # STRATEGY (Bayesian/Active Selection)
            if self.strategy in ["UCB", "EI"]:
                # Retrain surrogate
                surrogate.fit(X_train_strat, y_train_strat)
                
                strat_available_idx = np.where(strat_pool_mask)[0]
                X_strat_pool = X_pool[strat_available_idx]
                
                # Predict mean and std across trees
                preds = np.zeros((len(surrogate.estimators_), len(X_strat_pool)))
                for k, tree in enumerate(surrogate.estimators_):
                    preds[k, :] = tree.predict(X_strat_pool)
                    
                mean_preds = preds.mean(axis=0)
                std_preds = preds.std(axis=0)
                
                # Acquisition Function: UCB
                beta = 2.0  # Exploration-exploitation tradeoff
                ucb = mean_preds + beta * std_preds
                
                best_pool_idx = np.argmax(ucb)
                chosen_strat = strat_available_idx[best_pool_idx]
            else:
                # Fallback to random
                strat_available_idx = np.where(strat_pool_mask)[0]
                chosen_strat = np.random.choice(strat_available_idx)
                
            X_train_strat.append(X_pool[chosen_strat])
            y_train_strat.append(true_y_pool[chosen_strat])
            strat_pool_mask[chosen_strat] = False
            
            if true_y_pool[chosen_strat] > current_strategy_best:
                current_strategy_best = float(true_y_pool[chosen_strat])
                
            if current_strategy_best >= true_max * 0.98 and found_max_d33_iter is None:
                found_max_d33_iter = i
                
            strategy_curve.append((i, round(float(current_strategy_best), 2)))
            baseline_curve.append((i, round(float(current_baseline_best), 2)))
            
            # Emit progress via callback
            if cb and i % max(1, self.total_budget // 10) == 0:
                await cb({
                    "type": "progress",
                    "step": i,
                    "total": self.total_budget,
                    "best_d33": round(float(current_strategy_best), 2),
                    "message": f"Iteration {i}: Surrogate selected next candidate. Current best d33 = {current_strategy_best:.1f}"
                })
            
            # Allow event loop yield
            await asyncio.sleep(0.01)
            
        efficiency = 0.0
        if found_max_d33_iter is not None:
            efficiency = ((self.total_budget - found_max_d33_iter) / self.total_budget) * 100
        else:
            found_max_d33_iter = self.total_budget
            
        final_result = {
            "strategy_curve": strategy_curve,
            "baseline_curve": baseline_curve,
            "efficiency_gain": round(efficiency, 1),
            "iterations_to_max": {
                "strategy": found_max_d33_iter,
                "baseline": self.total_budget
            },
            "final_max_d33": round(float(current_strategy_best), 2)
        }
        
        return final_result
