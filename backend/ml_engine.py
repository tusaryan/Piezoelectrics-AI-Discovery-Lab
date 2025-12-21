import pandas as pd
import numpy as np
import re
import json
import chemparse
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.compose import TransformedTargetRegressor
try:
    from model_config import PARAM_GRIDS
except ImportError:
    PARAM_GRIDS = {}
    print("Warning: model_config.py not found. Using empty grids.")

import joblib
import os
import matplotlib.pyplot as plt
import io
import base64

# Configuration
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

ALL_ELEMENTS = ['Ag', 'Al', 'B', 'Ba', 'Bi', 'C', 'Ca', 'Fe', 'Hf', 'Ho', 'K',
                'Li', 'Mn', 'Na', 'Nb', 'O', 'Pr', 'Sb', 'Sc', 'Sr', 'Ta', 'Ti',
                'Zn', 'Zr']

# (Atomic Mass, Atomic Radius, Electronegativity, Valence Electrons)
ELEMENT_PROPERTIES = {
    'Ag': [107.87, 144, 1.93, 1],
    'Al': [26.98, 143, 1.61, 3],
    'B':  [10.81, 82, 2.04, 3],
    'Ba': [137.33, 222, 0.89, 2],
    'Bi': [208.98, 156, 2.02, 5],
    'C':  [12.01, 77, 2.55, 4],
    'Ca': [40.08, 197, 1.00, 2],
    'Fe': [55.85, 126, 1.83, 2], # Assuming common state
    'Hf': [178.49, 159, 1.30, 4],
    'Ho': [164.93, 176, 1.23, 3],
    'K':  [39.10, 227, 0.82, 1],
    'Li': [6.94, 152, 0.98, 1],
    'Mn': [54.94, 127, 1.55, 2],
    'Na': [22.99, 186, 0.93, 1],
    'Nb': [92.91, 146, 1.60, 5],
    'O':  [16.00, 60, 3.44, 6],
    'Pr': [140.91, 182, 1.13, 3],
    'Sb': [121.76, 140, 2.05, 5],
    'Sc': [44.96, 162, 1.36, 3],
    'Sr': [87.62, 215, 0.95, 2],
    'Ta': [180.95, 146, 1.50, 5],
    'Ti': [47.87, 147, 1.54, 4],
    'Zn': [65.38, 134, 1.65, 2],
    'Zr': [91.22, 160, 1.33, 4]
}
ELEMENT_PROP_NAMES = ["Avg_AtomicMass", "Avg_AtomicRadius", "Avg_Electronegativity", "Avg_Valence"]

try:
    from training_manager import training_manager
except ImportError:
    training_manager = None

VALID_PARAMS = {
    # Tree-based
    "Random Forest": [
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", 
        "max_features", "bootstrap", "oob_score", "n_jobs", "random_state", "verbose", "ccp_alpha"
    ],
    "Gradient Boosting": [
        "n_estimators", "learning_rate", "max_depth", "min_samples_split", 
        "min_samples_leaf", "subsample", "max_features", "random_state", "loss", "alpha", "verbose"
    ],
    "XGBoost": [
        "n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", 
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "objective", 
        "n_jobs", "random_state", "booster", "verbosity", "scale_pos_weight"
    ],
    "LightGBM": [
        "n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples", 
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "n_jobs", "random_state", "verbose"
    ],
    "SVM (SVR)": [
        "C", "epsilon", "kernel", "gamma", "degree", "coef0", "shrinking", "tol", "max_iter", "verbose"
    ],
    "Kernel Ridge": [
        "alpha", "kernel", "gamma", "degree", "coef0"
    ],
    "Gaussian Process": [
        "alpha", "n_restarts_optimizer", "normalize_y", "random_state"
    ],
    "Ensemble (Stacking)": ["cv", "n_jobs", "passthrough"]
}

def sanitize_params(model_type, raw_params):
    """
    Filters raw_params to only include valid arguments for the specified model_type.
    Prevents 'unexpected keyword argument' errors when switching model types.
    """
    if not raw_params:
        return {}
        
    allowed_keys = VALID_PARAMS.get(model_type, [])
    
    clean_params = {}
    for k, v in raw_params.items():
        if "__" in k:
            k_stripped = k.split("__")[-1]
        else:
            k_stripped = k
            
        if k_stripped in allowed_keys:
            clean_params[k_stripped] = v
            
    return clean_params



def parse_formula(formula_str, valid_elements=ALL_ELEMENTS):
    """
    Parses complex solid solutions (e.g., 0.96(K0.5Na0.5)NbO3-0.04Bi0.5Na0.5TiO3).
    """
    total_composition = {el: 0.0 for el in valid_elements}

    if not isinstance(formula_str, str):
        return total_composition

    clean_str = formula_str.replace(" ", "")
    clean_str = clean_str.replace("[", "(").replace("]", ")")
    clean_str = re.sub(r'\.(?=[A-Z]|\()', '-', clean_str)
    parts = re.split(r'[\-\+]', clean_str)

    for part in parts:
        if not part: continue

        coeff_match = re.match(r"^([\d\.]+)", part)
        if coeff_match:
            multiplier = float(coeff_match.group(1))
            sub_formula = part[len(coeff_match.group(1)):]
        else:
            multiplier = 1.0
            sub_formula = part

        sub_formula = sub_formula.lstrip("*")
        if not sub_formula: continue

        try:
            part_composition = chemparse.parse_formula(sub_formula)
        except Exception:
            continue

        if part_composition:
            for el, amt in part_composition.items():
                el_clean = ''.join([i for i in el if not i.isdigit()])
                if el_clean in valid_elements:
                    total_composition[el_clean] += amt * multiplier

    return total_composition

def create_feature_matrix(formula_series):
    """Creates feature matrix from formula series with Physics-Based Features."""
    feature_data = []
    physics_data = []
    
    for formula in formula_series:
        composition_dict = parse_formula(str(formula))
        feature_data.append(composition_dict)
        
        avg_props = [0.0, 0.0, 0.0, 0.0]
        total_atoms = 0.0
        
        for el, amt in composition_dict.items():
            if el in ELEMENT_PROPERTIES:
                props = ELEMENT_PROPERTIES[el]
                total_atoms += amt
                avg_props[0] += amt * props[0]
                avg_props[1] += amt * props[1]
                avg_props[2] += amt * props[2]
                avg_props[3] += amt * props[3]
        
        if total_atoms > 0:
            avg_props = [p / total_atoms for p in avg_props]
        
        physics_data.append(avg_props)
    
    if isinstance(formula_series, list):
        idx = range(len(formula_series))
    else:
        idx = formula_series.index
        
    df_features = pd.DataFrame(feature_data, columns=ALL_ELEMENTS, index=idx).fillna(0.0)
    
    df_physics = pd.DataFrame(physics_data, columns=ELEMENT_PROP_NAMES, index=idx)
    
    return pd.concat([df_features, df_physics], axis=1)

from sklearn.neighbors import NearestNeighbors

class AdvancedImputer:
    """
    Imputes missing target values based on chemical similarity (KNN).
    Uses the composition vectors to find most similar materials in the training set.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knowledge_base_X = None
        self.knowledge_base_y = None

    def fit(self, X_train, y_train):
        valid_mask = ~np.isnan(y_train)
        self.knowledge_base_X = X_train[valid_mask].copy()
        self.knowledge_base_y = y_train[valid_mask].copy()
        
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        self.nn_model.fit(self.knowledge_base_X)
        return self

    def transform(self, X, y):
        X_trans = X.copy()
        y_trans = y.copy()
        
        missing_mask = np.isnan(y_trans)
        if not np.any(missing_mask):
            return X_trans, y_trans
            
        X_missing = X_trans[missing_mask]
        distances, indices = self.nn_model.kneighbors(X_missing)
        
        weights = 1.0 / (distances + 1e-6)
        
        imputed_values = []
        for i in range(len(X_missing)):
            neighbor_indices = indices[i]
            neighbor_weights = weights[i]
            neighbor_values = self.knowledge_base_y.iloc[neighbor_indices].values
            
            val = np.sum(neighbor_weights * neighbor_values) / np.sum(neighbor_weights)
            imputed_values.append(val)
            
        y_trans[missing_mask] = imputed_values
        return X_trans, y_trans

import re

def clean_dataset(df, target_name, log_callback=None):
    """
    Cleans the dataset:
    - Removes rows with empty/NaN 'Component'.
    - Cleans target column (removes non-numeric chars like '$', 'd') and converts to float.
    - Logs dataset shape and head before/after.
    """
    def log(msg):
        if log_callback: log_callback(msg)
        print(msg)

    log(f"--- Cleaning Dataset for {target_name} ---")
    log(f"Initial Shape: {df.shape}")
    log(f"Initial Preview:\n{df.head().to_string()}")

    df_clean = df.dropna(subset=['Component']).copy()
    dropped_comp = len(df) - len(df_clean)
    if dropped_comp > 0:
        log(f"Dropped {dropped_comp} rows with missing 'Component'.")

    if target_name == "d33":
        col = 'd33 (pC/N)'
    else:
        col = next((c for c in df.columns if 'Tc' in c), None)

    if col and col in df_clean.columns:
        original_missing = df_clean[col].isna().sum()
        
        def clean_value(val):
            if pd.isna(val): return np.nan
            s = str(val)
            s_clean = re.sub(r'[^\d.-]', '', s)
            try:
                return float(s_clean)
            except ValueError:
                return np.nan

        df_clean[col] = df_clean[col].apply(clean_value)
        new_missing = df_clean[col].isna().sum()
        
        if new_missing > original_missing:
            log(f"Converted {new_missing - original_missing} corrupt target values to NaN for Imputation.")
    
    log(f"Cleaned Shape: {df_clean.shape}")
    log(f"Cleaned Preview:\n{df_clean.head().to_string()}")
    return df_clean

def check_interruption():
    if training_manager:
        training_manager.check_interruption()


def compare_models(X, y, target_name, manual_config=None, log_callback=None, components=None, training_mode="standard"):
    """Compares multiple models and returns metrics, plot data, and detailed logs."""
    def log(msg):
        if log_callback: log_callback(msg)
        print(msg)
    
    log(f"--- Starting Comparison for {target_name} ---")
    log(f"Dataset Shape: Features {X.shape}, Target {y.shape}")
    
    # 1. Strict Train/Test Split (80/20) BEFORE any processing on values
    if components is not None:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw, comp_train_raw, comp_test_raw = train_test_split(
            X, y, components, test_size=0.2, random_state=42
        )
    else:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
        comp_train_raw, comp_test_raw = None, None

    log(f"Data Split: Train {X_train_raw.shape[0]}, Test {X_test_raw.shape[0]}")
    
    train_missing = y_train_raw.isna().sum()
    test_missing = y_test_raw.isna().sum()
    log(f"Missing Targets (NaN): Train: {train_missing}, Test: {test_missing}")

    if train_missing > 0:
        log("Applying Advanced KNN Imputation on Training Set...")
        imputer = AdvancedImputer(n_neighbors=3)
        imputer.fit(X_train_raw, y_train_raw)
        X_train, y_train = imputer.transform(X_train_raw, y_train_raw)
        comp_train = comp_train_raw 
        log(f"Imputed {train_missing} values in Training Set using Chemical Similarity.")
    else:
        X_train, y_train = X_train_raw, y_train_raw
        comp_train = comp_train_raw
        log("No missing values in Train set. Skipping imputation.")

    if test_missing > 0:
        log(f"Dropping {test_missing} rows from Test Set with missing targets (Strict Evaluation).")
        valid_test_mask = ~y_test_raw.isna()
        X_test = X_test_raw[valid_test_mask]
        y_test = y_test_raw[valid_test_mask]
        comp_test = comp_test_raw[valid_test_mask] if comp_test_raw is not None else None
    else:
        X_test, y_test = X_test_raw, y_test_raw
        comp_test = comp_test_raw

    log(f"Final Processing Data: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    
    if not os.path.exists("debug_data"):
        os.makedirs("debug_data")
        
    train_objs = []
    if comp_train is not None: train_objs.append(comp_train)
    train_objs.append(X_train)
    train_objs.append(y_train)
    
    test_objs = []
    if comp_test is not None: test_objs.append(comp_test)
    test_objs.append(X_test)
    test_objs.append(y_test)
    
    train_df = pd.concat(train_objs, axis=1)
    test_df = pd.concat(test_objs, axis=1)
    
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()
    
    train_df.to_csv(f"debug_data/processed_train_data_{target_name}.csv", index=False)
    test_df.to_csv(f"debug_data/processed_test_data_{target_name}.csv", index=False)
    log(f"Saved processed datasets to 'debug_data/' for verification (Sorted by original index).")

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(verbose=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVM (SVR)": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1))
    }

    if training_mode == "accuracy":
        log("Activating Maximum Accuracy Mode: Deep Nested Optimization & Target Scaling...")
        
        rf_params = PARAM_GRIDS.get("Random Forest", {'n_estimators': [100]})
        xgb_params = PARAM_GRIDS.get("XGBoost", {'n_estimators': [100]})
        lgbm_params = PARAM_GRIDS.get("LightGBM", {'n_estimators': [100]})
        gb_params = PARAM_GRIDS.get("Gradient Boosting", {'n_estimators': [100]})
        svr_params = PARAM_GRIDS.get("SVM (SVR)", {'regressor__C': [1]})
        gpr_params = PARAM_GRIDS.get("Gaussian Process", {'regressor__n_restarts_optimizer': [0]})
        krr_params = PARAM_GRIDS.get("Kernel Ridge", {'regressor__alpha': [1.0]})
        
        gpr_kernel = np.var(y_train) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        
        models = {}
        
        models["Random Forest"] = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=1), 
            rf_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
        
        models["XGBoost"] = GridSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42),
            xgb_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
        
        models["LightGBM"] = GridSearchCV(
            lgb.LGBMRegressor(verbose=-1, random_state=42, n_jobs=1),
            lgbm_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
        
        models["Gradient Boosting"] = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
        
        models["SVM (SVR)"] = GridSearchCV(
            TransformedTargetRegressor(
                regressor=SVR(kernel='rbf'), 
                transformer=StandardScaler()
            ),
            svr_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
        
        models["Gaussian Process"] = GridSearchCV(
            TransformedTargetRegressor(
                regressor=GaussianProcessRegressor(kernel=gpr_kernel, normalize_y=True), 
                transformer=StandardScaler()
            ),
            gpr_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
        
        models["Kernel Ridge"] = GridSearchCV(
            TransformedTargetRegressor(
                regressor=KernelRidge(kernel='rbf'), 
                transformer=StandardScaler()
            ),
            krr_params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
        )
    # Manual Configuration
    if manual_config and manual_config.get("model_type"):
        model_type = manual_config["model_type"]
        raw_params = manual_config.get("params", {})
        
        params = sanitize_params(model_type, raw_params)
        
        log(f"Manual Configuration Detected: Overwriting {model_type} with fixed parameters: {params}")
        
        if model_type == "Random Forest":
            models[model_type] = RandomForestRegressor(random_state=42, **params)
        elif model_type == "XGBoost":
            models[model_type] = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **params)
        elif model_type == "LightGBM":
            models[model_type] = lgb.LGBMRegressor(verbose=-1, random_state=42, **params)
        elif model_type == "Gradient Boosting":
            models[model_type] = GradientBoostingRegressor(random_state=42, **params)
        elif model_type == "SVM (SVR)":
            models[model_type] = make_pipeline(StandardScaler(), SVR(**params))
        elif model_type == "Gaussian Process":
            gpr_kernel = np.var(y_train) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
            models[model_type] = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=gpr_kernel, normalize_y=True, **params))
        elif model_type == "Kernel Ridge":
            models[model_type] = make_pipeline(StandardScaler(), KernelRidge(**params))

    results = []
    predictions = {}
    
    fitted_estimators = {}
    
    for name, model in models.items():
        check_interruption()
        log(f"Training {name} (Nested Optimization)..." if training_mode == "accuracy" else f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"Model": name, "R2": r2, "RMSE": rmse})
        predictions[name] = {"y_test": y_test.tolist(), "y_pred": y_pred.tolist()}
        
        if training_mode == "accuracy":
            fitted_estimators[name] = model.best_estimator_

    if training_mode == "accuracy":
        log("Building Stacking Ensemble from Optimized Models...")
        check_interruption()
        
        stacking_estimators = [
            ('rf', fitted_estimators["Random Forest"]),
            ('xgb', fitted_estimators["XGBoost"]),
            ('lgbm', fitted_estimators["LightGBM"]),
            ('gb', fitted_estimators["Gradient Boosting"]),
            ('svr', fitted_estimators["SVM (SVR)"]),
            ('gpr', fitted_estimators["Gaussian Process"]),
            ('krr', fitted_estimators["Kernel Ridge"])
        ]
        
        ensemble = StackingRegressor(
            estimators=stacking_estimators,
            final_estimator=GradientBoostingRegressor(n_estimators=50, subsample=0.5, random_state=42),
            n_jobs=-1,
            cv=3
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({"Model": "Ensemble (Stacking)", "R2": r2, "RMSE": rmse})
        predictions["Ensemble (Stacking)"] = {"y_test": y_test.tolist(), "y_pred": y_pred.tolist()}

    return results, predictions

def train_production_model(X, y, target_name, model_type="Auto", params=None, auto_tune=False, training_mode="standard", save_as_temp=False):
    """
    Trains the selected model for production use.
    Uses full dataset for training.
    """
    model_filename = f"{target_name}_model.pkl"
    if save_as_temp:
        model_filename += ".tmp"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    best_model = None
    feature_importance = {}
    
    if training_mode == "accuracy":
        pass
    
    check_interruption() 
    
    if auto_tune:

        base_model = None
        search_params = {}
        
        if training_mode == "accuracy":
            if model_type == "XGBoost":
                base_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
                search_params = {
                    'n_estimators': [100, 300, 500, 700, 1000],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 5, 7, 9, 11, 15],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [1, 5, 10, 50]
                }
            elif model_type == "Random Forest":
                base_model = RandomForestRegressor(random_state=42)
                search_params = {
                    'n_estimators': [100, 300, 500, 800, 1000],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif model_type == "LightGBM":
                base_model = lgb.LGBMRegressor(verbose=-1, random_state=42)
                search_params = {
                    'n_estimators': [100, 300, 500, 1000],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [20, 31, 50, 70, 100, 150],
                    'max_depth': [-1, 10, 20, 30],
                    'min_child_samples': [10, 20, 30, 50],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                }
            elif model_type == "Gradient Boosting":
                base_model = GradientBoostingRegressor(random_state=42)
                search_params = {
                    'n_estimators': [100, 300, 500, 800],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == "SVM (SVR)":
                base_model = make_pipeline(StandardScaler(), SVR())
                search_params = {
                    'svr__C': [0.1, 1, 10, 50, 100, 500, 1000],
                    'svr__epsilon': [0.001, 0.01, 0.1, 0.2, 0.5],
                    'svr__kernel': ['rbf', 'linear', 'poly'],
                    'svr__gamma': ['scale', 'auto', 0.01, 0.1, 1]
                }
        if training_mode == "accuracy":
            
            # 1. Random Forest
            if model_type == "Random Forest":
                base_model = RandomForestRegressor(random_state=42, n_jobs=1)
                search_params = PARAM_GRIDS.get("Random Forest", {'n_estimators': [100]})
            
            # 2. XGBoost
            elif model_type == "XGBoost":
                base_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42)
                search_params = PARAM_GRIDS.get("XGBoost", {'n_estimators': [100]})
                
            # 3. LightGBM
            elif model_type == "LightGBM":
                base_model = lgb.LGBMRegressor(verbose=-1, random_state=42, n_jobs=1)
                search_params = PARAM_GRIDS.get("LightGBM", {'n_estimators': [100]})
                
            # 4. Gradient Boosting
            elif model_type == "Gradient Boosting":
                base_model = GradientBoostingRegressor(random_state=42)
                search_params = PARAM_GRIDS.get("Gradient Boosting", {'n_estimators': [100]})
                
            # 5. SVM (SVR)
            elif model_type == "SVM (SVR)":
                base_model = TransformedTargetRegressor(
                    regressor=SVR(),
                    transformer=StandardScaler()
                )
                search_params = PARAM_GRIDS.get("SVM (SVR)", {'regressor__C': [1.0]})
                
            # 6. Gaussian Process
            elif model_type == "Gaussian Process":
                kernel = np.var(y) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
                base_model = TransformedTargetRegressor(
                    regressor=GaussianProcessRegressor(kernel=kernel, normalize_y=True),
                    transformer=StandardScaler()
                )
                search_params = PARAM_GRIDS.get("Gaussian Process", {'regressor__n_restarts_optimizer': [0]})
                
            # 7. Kernel Ridge
            elif model_type == "Kernel Ridge":
                base_model = TransformedTargetRegressor(
                    regressor=KernelRidge(kernel='rbf'),
                    transformer=StandardScaler()
                )
                search_params = PARAM_GRIDS.get("Kernel Ridge", {'regressor__alpha': [1.0]})
            
            # 8. Ensemble (Stacking)
            elif model_type == "Ensemble (Stacking)":
                 rf = GridSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=1), 
                    PARAM_GRIDS.get("Random Forest", {'n_estimators': [100]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 xgb_est = GridSearchCV(
                    xgb.XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42),
                    PARAM_GRIDS.get("XGBoost", {'n_estimators': [100]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 lgbm = GridSearchCV(
                    lgb.LGBMRegressor(verbose=-1, random_state=42, n_jobs=1),
                    PARAM_GRIDS.get("LightGBM", {'n_estimators': [100]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 gb = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    PARAM_GRIDS.get("Gradient Boosting", {'n_estimators': [100]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 svr = GridSearchCV(
                    TransformedTargetRegressor(regressor=SVR(kernel='rbf'), transformer=StandardScaler()),
                    PARAM_GRIDS.get("SVM (SVR)", {'regressor__C': [1]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 gpr_kernel = np.var(y) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
                 gpr = GridSearchCV(
                    TransformedTargetRegressor(regressor=GaussianProcessRegressor(kernel=gpr_kernel, normalize_y=True), transformer=StandardScaler()),
                    PARAM_GRIDS.get("Gaussian Process", {'regressor__n_restarts_optimizer': [0]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 krr = GridSearchCV(
                    TransformedTargetRegressor(regressor=KernelRidge(kernel='rbf'), transformer=StandardScaler()),
                    PARAM_GRIDS.get("Kernel Ridge", {'regressor__alpha': [1.0]}), 
                    cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error'
                 )
                 
                 estimators = [
                    ('rf', rf), ('xgb', xgb_est), ('lgbm', lgbm), ('gb', gb), ('svr', svr), ('gpr', gpr), ('krr', krr)
                 ]
                 
                 # 2. Define Stacking Regressor
                 base_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=GradientBoostingRegressor(n_estimators=50, subsample=0.5, random_state=42),
                    n_jobs=-1,
                    cv=3
                 )
                 search_params = {} 

        else:
            if model_type == "Random Forest":
                base_model = RandomForestRegressor(random_state=42, n_jobs=1)
                search_params = {
                    'n_estimators': [100, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == "XGBoost":
                base_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42)
                search_params = {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            elif model_type == "LightGBM":
                base_model = lgb.LGBMRegressor(verbose=-1, random_state=42, n_jobs=1)
                search_params = {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50]
                }
            elif model_type == "Gradient Boosting":
                base_model = GradientBoostingRegressor(random_state=42)
                search_params = {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            elif model_type == "SVM (SVR)":
                base_model = make_pipeline(StandardScaler(), SVR())
                search_params = {
                    'svr__C': [0.1, 1, 10, 100],
                    'svr__epsilon': [0.01, 0.1, 0.2],
                    'svr__kernel': ['rbf', 'linear']
                }
            elif model_type == "Ensemble (Stacking)":
                estimators = [
                    ('rf', RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42)),
                    ('xgb', xgb.XGBRegressor(n_estimators=100, n_jobs=1, random_state=42)),
                    ('svr', make_pipeline(StandardScaler(), SVR(C=100)))
                ]
                base_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=GradientBoostingRegressor(n_estimators=50, random_state=42),
                    n_jobs=-1,
                    cv=3
                )
                search_params = {}

        cv_folds = 5 if training_mode == "accuracy" else 3
        #fast
        if base_model is not None:
            check_interruption()
            
            if training_mode == "accuracy":
                 search = GridSearchCV(base_model, search_params, scoring='neg_root_mean_squared_error', cv=cv_folds, verbose=0, n_jobs=-1)
            else:
                 search = RandomizedSearchCV(base_model, search_params, n_iter=10, scoring='neg_root_mean_squared_error', cv=cv_folds, verbose=0, random_state=42, n_jobs=-1)
            
            search.fit(X, y)
            best_model = search.best_estimator_

        else:
             best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
             best_model.fit(X, y)

    
    else:
        raw_params = params or {}
        clean_params = sanitize_params(model_type, raw_params)
        
        if model_type == "Random Forest":
            best_model = RandomForestRegressor(random_state=42, **clean_params)
        elif model_type == "XGBoost":
            best_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **clean_params)
        elif model_type == "LightGBM":
            best_model = lgb.LGBMRegressor(verbose=-1, random_state=42, **clean_params)
        elif model_type == "Gradient Boosting":
            best_model = GradientBoostingRegressor(random_state=42, **clean_params)
        elif model_type == "SVM (SVR)":
            best_model = make_pipeline(StandardScaler(), SVR(**clean_params))
        else:
            best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

        check_interruption()
        best_model.fit(X, y)


    # Extract Feature Importance
    try:
        if hasattr(best_model, "feature_importances_"):
            imps = best_model.feature_importances_
            feats = X.columns
            imp_dict = {f: float(i) for f, i in zip(feats, imps)}
            feature_importance = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)[:10])
            
        elif hasattr(best_model, "named_steps") and "svr" in best_model.named_steps:
             svr = best_model.named_steps["svr"]
             if svr.kernel == 'linear' and hasattr(svr, "coef_"):
                 imps = np.abs(svr.coef_[0])
                 feats = X.columns
                 imp_dict = {f: float(i) for f, i in zip(feats, imps)}
                 feature_importance = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    joblib.dump(best_model, model_path)
    return best_model, feature_importance

def save_active_model_info(d33_model_name, d33_mode, tc_model_name, tc_mode):
    """Saves active model info to disk."""
    info = {
        "d33": {"name": d33_model_name, "mode": d33_mode},
        "Tc": {"name": tc_model_name, "mode": tc_mode}
    }
    with open(os.path.join(MODEL_DIR, "active_model_info.json"), "w") as f:
        json.dump(info, f)

def load_active_model_info():
    """Loads active model info from disk."""
    try:
        with open(os.path.join(MODEL_DIR, "active_model_info.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "d33": {"name": "Unknown", "mode": "Unknown"},
            "Tc": {"name": "Unknown", "mode": "Unknown"}

        }

def finalize_staging_models():
    """Renames .tmp model files to production files."""
    for target in ["d33", "Tc"]:
        tmp_path = os.path.join(MODEL_DIR, f"{target}_model.pkl.tmp")
        prod_path = os.path.join(MODEL_DIR, f"{target}_model.pkl")
        if os.path.exists(tmp_path):
            if os.path.exists(prod_path):
                os.remove(prod_path)
            os.rename(tmp_path, prod_path)

def predict_properties(formula):
    """Predicts d33 and Tc for a given formula."""
    try:
        model_d33 = joblib.load(os.path.join(MODEL_DIR, "d33_model.pkl"))
        model_tc = joblib.load(os.path.join(MODEL_DIR, "Tc_model.pkl"))
    except FileNotFoundError:
        return None, None, None

    # Parse and Featurize
    composition = parse_formula(formula)
    X = create_feature_matrix([formula])
    
    # Predict
    try:
        if hasattr(X, 'empty') and X.empty:
             return None, None, composition
             
        d33_pred = model_d33.predict(X)[0]
        tc_pred = model_tc.predict(X)[0]
        
        return float(d33_pred), float(tc_pred), composition
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None, None, composition



def save_insights(results_d33=None, predictions_d33=None, results_tc=None, predictions_tc=None, feature_importance_d33=None, feature_importance_tc=None):
    """Saves insights to disk."""
    insights = {
        "comparison_d33": results_d33 or [],
        "comparison_tc": results_tc or [],
        "predictions_d33": predictions_d33 or {},
        "predictions_tc": predictions_tc or {},
        "scatter_d33": [],
        "scatter_tc": [],
        "feature_importance_d33": feature_importance_d33 or {},
        "feature_importance_tc": feature_importance_tc or {}
    }
    
    if predictions_d33:
        first_model = list(predictions_d33.keys())[0]
        y_test = predictions_d33[first_model]['y_test']
        y_pred = predictions_d33[first_model]['y_pred']
        insights["scatter_d33"] = [{"x": float(act), "y": float(pred)} for act, pred in zip(y_test, y_pred)]

    if predictions_tc:
        first_model = list(predictions_tc.keys())[0]
        y_test = predictions_tc[first_model]['y_test']
        y_pred = predictions_tc[first_model]['y_pred']
        insights["scatter_tc"] = [{"x": float(act), "y": float(pred)} for act, pred in zip(y_test, y_pred)]

    joblib.dump(insights, os.path.join(MODEL_DIR, "insights.pkl"))

def load_insights():
    """Loads insights from disk."""
    try:
        return joblib.load(os.path.join(MODEL_DIR, "insights.pkl"))
    except FileNotFoundError:
        return {
            "comparison_d33": [], 
            "comparison_tc": [], 
            "scatter_d33": [], 
            "scatter_tc": []
        }
