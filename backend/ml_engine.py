import pandas as pd
import numpy as np
import re
import json
import chemparse
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

def parse_formula(formula_str, valid_elements=ALL_ELEMENTS):
    """
    Parses complex solid solutions (e.g., 0.96(K0.5Na0.5)NbO3-0.04Bi0.5Na0.5TiO3).
    """
    total_composition = {el: 0.0 for el in valid_elements}

    if not isinstance(formula_str, str):
        return total_composition

    # Normalization
    clean_str = formula_str.replace(" ", "")
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
    """Creates feature matrix from formula series."""
    feature_data = []
    for formula in formula_series:
        composition_dict = parse_formula(str(formula))
        feature_data.append(composition_dict)
    
    df_features = pd.DataFrame(feature_data, columns=ALL_ELEMENTS)
    return df_features.fillna(0.0)

def compare_models(X, y, target_name, manual_config=None):
    """Compares multiple models and returns metrics and plot data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(verbose=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVM (SVR)": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1))
    }

    # Override with manual config if provided
    if manual_config and manual_config.get("model_type") in models:
        model_type = manual_config["model_type"]
        params = manual_config.get("params", {})
        
        if model_type == "Random Forest":
            models[model_type] = RandomForestRegressor(random_state=42, **params)
        elif model_type == "XGBoost":
            models[model_type] = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **params)
        elif model_type == "LightGBM":
            models[model_type] = lgb.LGBMRegressor(verbose=-1, random_state=42, **params)
        elif model_type == "Gradient Boosting":
            models[model_type] = GradientBoostingRegressor(random_state=42, **params)
        elif model_type == "SVM (SVR)":
            # SVR needs pipeline for scaling, so we reconstruct it
            models[model_type] = make_pipeline(StandardScaler(), SVR(**params))
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"Model": name, "R2": r2, "RMSE": rmse})
        predictions[name] = {"y_test": y_test.tolist(), "y_pred": y_pred.tolist()}
        
    return results, predictions

def train_production_model(X, y, target_name, model_type="Auto", params=None):
    """Trains and saves the production model."""
    model_path = os.path.join(MODEL_DIR, f"final_model_{target_name}.pkl")
    
    if model_type == "Auto":
        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.6, 0.8]
        }
        search = RandomizedSearchCV(base_model, param_dist, n_iter=10, scoring='neg_root_mean_squared_error', cv=3, verbose=0, random_state=42)
        search.fit(X, y)
        best_model = search.best_estimator_
    else:
        # Manual model selection
        if model_type == "Random Forest":
            best_model = RandomForestRegressor(random_state=42, **(params or {}))
        elif model_type == "XGBoost":
            best_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **(params or {}))
        elif model_type == "LightGBM":
            best_model = lgb.LGBMRegressor(verbose=-1, random_state=42, **(params or {}))
        elif model_type == "Gradient Boosting":
            best_model = GradientBoostingRegressor(random_state=42, **(params or {}))
        elif model_type == "SVM (SVR)":
            best_model = make_pipeline(StandardScaler(), SVR(**(params or {})))
        else:
            # Fallback
            best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

        best_model.fit(X, y)

    joblib.dump(best_model, model_path)
    return best_model

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

def predict_properties(formula):
    """Predicts d33 and Tc for a given formula."""
    try:
        # Load models
        model_d33 = joblib.load(os.path.join(MODEL_DIR, "final_model_d33.pkl"))
        model_tc = joblib.load(os.path.join(MODEL_DIR, "final_model_Tc.pkl"))
    except FileNotFoundError:
        return None, None, None

    composition = parse_formula(formula)
    input_df = pd.DataFrame([composition], columns=ALL_ELEMENTS).fillna(0.0)
    
    d33 = model_d33.predict(input_df)[0]
    tc = model_tc.predict(input_df)[0]
    
    return float(d33), float(tc), composition

def save_insights(results_d33=None, predictions_d33=None, results_tc=None, predictions_tc=None):
    """Saves insights to disk."""
    insights = {
        "comparison_d33": results_d33 or [],
        "comparison_tc": results_tc or [],
        "scatter_d33": [],
        "scatter_tc": []
    }
    
    # Format scatter data for d33
    if predictions_d33:
        first_model = list(predictions_d33.keys())[0]
        y_test = predictions_d33[first_model]['y_test']
        y_pred = predictions_d33[first_model]['y_pred']
        insights["scatter_d33"] = [{"x": float(act), "y": float(pred)} for act, pred in zip(y_test, y_pred)]

    # Format scatter data for Tc
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
