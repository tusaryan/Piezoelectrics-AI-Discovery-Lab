import logging
from typing import Dict, Any, Optional
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

logger = logging.getLogger(__name__)

class CompositeTrainer:
    """
    Dedicated training pipeline for Piezocomposite materials.
    Focuses specifically on predicting 'composite_d33' using the extended 43-dimensional vector.
    """
    def __init__(self, data_loder_ref: Any = None):
        # Mocks the data loader reference for now
        self.data_loader = data_loder_ref
        
    def train_composite_models(self, X: np.ndarray, y_composite_d33: np.ndarray, log_callback=None) -> Dict[str, Any]:
        """
        Trains XGBoost and ANN on composite data.
        """
        if len(y_composite_d33) < 10:
            if log_callback:
                log_callback("WARNING", "Insufficient composite data points (<10) to train a reliable model.", 0)
            return {"status": "skipped", "reason": "insufficient_data"}
            
        # Optional subsetting if dealing with mixed data arrays
        valid_idx = ~np.isnan(y_composite_d33)
        X_clean = X[valid_idx]
        y_clean = y_composite_d33[valid_idx]
        
        if len(y_clean) < 10:
             return {"status": "skipped", "reason": "insufficient_valid_data"}
        
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        
        if log_callback:
            log_callback("INFO", f"Starting training on {len(X_train)} piezocomposites...", 1)
            
        # 1. XGBoost
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_preds)
        xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
        
        if log_callback:
            log_callback("INFO", f"XGBoost finished. R2: {xgb_r2:.3f}, RMSE: {xgb_rmse:.2f}", 2)
            
        # 2. ANN
        ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        ann.fit(X_train, y_train)
        ann_preds = ann.predict(X_test)
        ann_r2 = r2_score(y_test, ann_preds)
        ann_rmse = root_mean_squared_error(y_test, ann_preds)
        
        if log_callback:
            log_callback("INFO", f"ANN finished. R2: {ann_r2:.3f}, RMSE: {ann_rmse:.2f}", 3)
            
        best_model = "xgboost" if xgb_r2 > ann_r2 else "ann"
        
        if log_callback:
            log_callback("INFO", f"Composite Training Complete. Best model selected: {best_model.upper()}", 4)

        return {
            "status": "success",
            "best_model": best_model,
            "metrics": {
                "xgboost": {"r2": float(xgb_r2), "rmse": float(xgb_rmse)},
                "ann": {"r2": float(ann_r2), "rmse": float(ann_rmse)}
            },
            "models": {
                "xgboost": xgb,
                "ann": ann
            }
        }
