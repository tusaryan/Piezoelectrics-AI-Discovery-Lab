import logging
from typing import Dict, Any
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

logger = logging.getLogger(__name__)

class HardnessTrainer:
    """
    Dedicated training pipeline targeting vickers_hardness and mohs_hardness.
    """
    def __init__(self, data_loader_ref: Any = None):
        self.data_loader = data_loader_ref

    def train_models(self, X: np.ndarray, y_vickers: np.ndarray, y_mohs: np.ndarray, log_callback=None) -> Dict[str, Any]:
        """
        Trains GBR + XGBoost on mechanical strength targets.
        """
        # Filter valid rows for training
        valid_idx = ~np.isnan(y_vickers)
        X_clean = X[valid_idx]
        y_clean = y_vickers[valid_idx]

        if len(y_clean) < 10:
             if log_callback: log_callback("WARNING", "Insufficient vickers_hardness data.", 0)
             return {"status": "skipped"}
             
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        
        # Train Vickers XGBoost
        xgb_v = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        xgb_v.fit(X_train, y_train)
        pred_v = xgb_v.predict(X_test)
        r2_v = r2_score(y_test, pred_v)
        
        if log_callback:
            log_callback("INFO", f"Vickers Hardness (HV) Model trained. R2: {r2_v:.3f}", 1)

        # Attempt Mohs if data exists
        valid_mohs = ~np.isnan(y_mohs)
        X_mohs = X[valid_mohs]
        y_mohs_clean = y_mohs[valid_mohs]
        
        xgb_m = None
        r2_m = 0.0
        if len(y_mohs_clean) > 10:
             X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mohs, y_mohs_clean, test_size=0.2, random_state=42)
             xgb_m = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
             xgb_m.fit(X_train_m, y_train_m)
             pred_m = xgb_m.predict(X_test_m)
             r2_m = r2_score(y_test_m, pred_m)
             if log_callback:
                 log_callback("INFO", f"Mohs Hardness Model trained. R2: {r2_m:.3f}", 2)

        return {
            "status": "success",
            "models": {
                "vickers_xgb": xgb_v,
                "mohs_xgb": xgb_m
            },
            "metrics": {
                "vickers_r2": float(r2_v),
                "mohs_r2": float(r2_m)
            }
        }
