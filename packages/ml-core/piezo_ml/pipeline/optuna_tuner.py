import optuna
import importlib
import numpy as np
from typing import Dict, Any, Callable
from sklearn.model_selection import cross_val_score
from piezo_ml.models.registry import MODEL_REGISTRY

class OptunaTuner:
    """
    Optuna hyperparameter tuning engine executing within the training pipeline.
    """
    
    def __init__(self, model_name: str, x_train: np.ndarray, y_train: np.ndarray, log_callback: Callable):
        self.model_name = model_name
        self.x_train = x_train
        self.y_train = y_train
        self.log_callback = log_callback
        self.schema = MODEL_REGISTRY.get(model_name)
        if not self.schema:
            raise ValueError(f"Model {model_name} not found in registry")
            
        params_array = self.schema["class_name"].split(".")
        module_name = ".".join(params_array[:-1])
        class_name = params_array[-1]
        
        module = importlib.import_module(module_name)
        self.model_class = getattr(module, class_name)
        
    def _objective(self, trial) -> float:
        params = {}
        for param in self.schema["params"]:
            name = param["name"]
            p_type = param["type"]
            range_min, range_max = param["optuna_range"]
            
            if p_type == "int":
                params[name] = trial.suggest_int(name, range_min, range_max)
            elif p_type == "float_log":
                params[name] = trial.suggest_float(name, range_min, range_max, log=True)
            elif p_type == "float":
                params[name] = trial.suggest_float(name, range_min, range_max)
                
        # Specialized fixes
        if self.model_name == "RandomForest" and params.get("max_depth", 0) == 0:
            params["max_depth"] = None
            
        model = self.model_class(**params)
        
        # 3-fold CV R2 scoring
        scores = cross_val_score(model, self.x_train, self.y_train, cv=3, scoring='r2')
        mean_r2 = scores.mean()
        
        self.log_callback("INFO", f"Optuna Trial {trial.number} finished", "optuna", {
            "params": params,
            "r2": float(mean_r2)
        })
        
        return mean_r2
        
    def tune(self, n_trials: int = 50) -> Dict[str, Any]:
        self.log_callback("INFO", f"Starting Optuna optimization for {self.model_name}", "optuna_start", {"n_trials": n_trials})
        
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner()
        
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=n_trials)
        
        best_params = study.best_params
        # Optuna changes RandomForest back to integers/None logic here if needed
        if self.model_name == "RandomForest" and best_params.get("max_depth", 0) == 0:
            best_params["max_depth"] = None
            
        self.log_callback("INFO", f"Optuna finished", "optuna_end", {"best_r2": study.best_value})
        
        return best_params
