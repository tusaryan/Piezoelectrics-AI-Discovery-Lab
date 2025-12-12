from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import io
import os
import json
from ml_engine import (
    predict_properties, ALL_ELEMENTS, save_insights, save_active_model_info, load_active_model_info,
    clean_dataset, finalize_staging_models, create_feature_matrix, train_production_model, compare_models
)
from report_generator import generate_report
from training_manager import training_manager

router = APIRouter()

# Check if models exist on startup
if os.path.exists("saved_models/d33_model.pkl") and os.path.exists("saved_models/Tc_model.pkl"):
    # We can't easily update the singleton's internal state on module load without a method, 
    # but the singleton defaults to False.
    # It's better to check existence in the status endpoint or predict endpoint dynamically if possible,
    # or add a method 'set_trained_flag' to manager.
    # For now, let's trust the file check in predict/status.
    pass

class PredictRequest(BaseModel):
    formula: str

class TrainRequest(BaseModel):
    model_type: str = "Auto"
    params: Optional[Dict] = None

@router.get("/status")
def get_status():
    state = training_manager.get_state()
    # Augment state with file check for 'is_trained' if not strictly tracked in memory yet
    if os.path.exists("saved_models/d33_model.pkl") and os.path.exists("saved_models/Tc_model.pkl"):
         state["is_trained"] = True
    else:
         state["is_trained"] = False
    return state

@router.post("/predict")
def predict(request: PredictRequest):
    state = training_manager.get_state()
    if state["status"] == "training":
        raise HTTPException(status_code=503, detail="Model is currently retraining. Please wait.")
    
    # Double check file existence
    if not (os.path.exists("saved_models/d33_model.pkl") and os.path.exists("saved_models/Tc_model.pkl")):
            raise HTTPException(status_code=400, detail="Models not trained yet. Please go to the Retraining tab.")

    d33, tc, composition = predict_properties(request.formula)
    if d33 is None:
        raise HTTPException(status_code=400, detail="Prediction failed.")
    
    # Filter out zero values for cleaner display
    active_composition = {k: v for k, v in composition.items() if v > 0}
    
    return {"d33": d33, "Tc": tc, "composition": active_composition}

@router.post("/stop-training")
def stop_training():
    state = training_manager.get_state()
    if state["status"] == "training":
        training_manager.abort_training()
        return {"message": "Stop signal sent."}
    return {"message": "No active training session."}

@router.post("/train")
async def train(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    model_type: str = Form("Auto"),
    auto_tune: bool = Form(False),
    n_estimators: int = Form(100),
    learning_rate: float = Form(0.1),
    max_depth: int = Form(5),
    c_param: float = Form(100.0),
    epsilon: float = Form(0.1),
    training_mode: str = Form("standard"),
    d33_model_type: str = Form(None),
    tc_model_type: str = Form(None),
    d33_params: str = Form(None), # JSON string
    tc_params: str = Form(None)   # JSON string
):
    state = training_manager.get_state()
    if state["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress.")

    # Read dataset
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file.")
    
    # Check columns
    if 'Component' not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset must have 'Component' column.")
        
    # Construct params dict based on legacy/global inputs if specific ones not provided
    global_params = {}
    if model_type != "Auto":
        if model_type in ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]:
            global_params["n_estimators"] = n_estimators
            global_params["max_depth"] = max_depth
        
        if model_type in ["XGBoost", "LightGBM", "Gradient Boosting"]:
            global_params["learning_rate"] = learning_rate
            
        if model_type == "SVM (SVR)":
            global_params["C"] = c_param
            global_params["epsilon"] = epsilon
            
    # Parse specific JSON params if available
    d33_params_dict = None
    tc_params_dict = None
    
    if d33_params:
        try:
            d33_params_dict = json.loads(d33_params)
        except:
            print("Failed to parse d33_params JSON")
            
    if tc_params:
        try:
            tc_params_dict = json.loads(tc_params)
        except:
            print("Failed to parse tc_params JSON")

    # Trigger training in background
    # IMPORTANT: Mark training as started synchronously to avoid race condition with frontend polling
    print("DEBUG: Calling training_manager.start_training()...")
    training_manager.start_training()
    print(f"DEBUG: State after start: {training_manager.get_state()}")
    
    background_tasks.add_task(run_training_pipeline, df, model_type, global_params, auto_tune, training_mode, d33_model_type, tc_model_type, d33_params_dict, tc_params_dict)
    
    return {"message": "Training started."}


# Import logger
from utils.logger import TrainingContext

def run_training_pipeline(df, model_type, params, auto_tune=False, training_mode="standard", d33_model_type=None, tc_model_type=None, d33_params=None, tc_params=None):
    # Adapter for existing ML functions that expect a callback
    log_callback = lambda msg: training_manager.log("info", msg)
    
    # Use Context Manager to redirct ALL print statements to logger
    with TrainingContext(training_manager.log):
        try:
            training_manager.check_interruption()
            
            # LOGGING INITIAL STATS
            print(f"Initial Dataset Size: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")

            # Pre-clean Datasets
            training_manager.update_progress(5, "Preprocessing Data...")
            
            df_d33_clean = None
            df_tc_clean = None
            
            if 'd33 (pC/N)' in df.columns:
                print("Cleaning d33 data...")
                df_d33_clean = clean_dataset(df, "d33", log_callback)
                print(f"Cleaned d33 Data: {len(df_d33_clean)} valid samples found.")
                
            training_manager.check_interruption()

            tc_col = next((c for c in df.columns if 'Tc' in c), None)
            if tc_col:
                print("Cleaning Tc data...")
                df_tc_clean = clean_dataset(df, "Tc", log_callback)
                print(f"Cleaned Tc Data: {len(df_tc_clean)} valid samples found.")

            training_manager.check_interruption()

            # ----------------------------------------------------
            # 1. COMPARE ALGORITHMS (Find the Best)
            # ----------------------------------------------------
            results_d33, predictions_d33 = None, None
            results_tc, predictions_tc = None, None
            
            # Decide Target Models
            target_d33 = d33_model_type if d33_model_type and d33_model_type != "Auto" else (model_type if model_type not in ["Auto", "Manual_Granular"] else "Auto")
            target_tc = tc_model_type if tc_model_type and tc_model_type != "Auto" else (model_type if model_type not in ["Auto", "Manual_Granular"] else "Auto")
            
            # Compare D33
            best_d33_model = target_d33 
            if df_d33_clean is not None and len(df_d33_clean) > 10:
                training_manager.update_progress(20, "Benchmarking models for d33...")
                X = create_feature_matrix(df_d33_clean['Component'])
                y = df_d33_clean['d33 (pC/N)']
                
                training_manager.check_interruption()
                
                # BUG FIX: Pass Manual Config to Comparison Report
                manual_config_d33 = None
                if target_d33 != "Auto":
                    final_d33_params = d33_params if d33_params else params
                    manual_config_d33 = {"model_type": target_d33, "params": final_d33_params or {}}
                
                results_d33, predictions_d33 = compare_models(
                    X, y, "d33", 
                    manual_config=manual_config_d33, 
                    log_callback=log_callback, 
                    components=df_d33_clean['Component'], 
                    training_mode=training_mode
                )
                
                if results_d33:
                    best_perf = sorted(results_d33, key=lambda x: x['R2'], reverse=True)[0]
                    auto_winner = best_perf['Model']
                    print(f"Auto-Tune identified best d33 model: {auto_winner} (R2: {best_perf['R2']:.4f})")
                    
                    if target_d33 == "Auto":
                        best_d33_model = auto_winner
                    else:
                        print(f"Using User Selected Model for d33: {target_d33}")
            
            training_manager.check_interruption()

            # Compare Tc
            best_tc_model = target_tc
            if df_tc_clean is not None and len(df_tc_clean) > 10:
                training_manager.update_progress(40, "Benchmarking models for Tc...")
                X = create_feature_matrix(df_tc_clean['Component'])
                y = df_tc_clean[tc_col]
                
                training_manager.check_interruption()
                
                # BUG FIX: Pass Manual Config to Comparison Report
                manual_config_tc = None
                if target_tc != "Auto":
                    final_tc_params = tc_params if tc_params else params
                    manual_config_tc = {"model_type": target_tc, "params": final_tc_params or {}}
                
                results_tc, predictions_tc = compare_models(
                    X, y, "Tc", 
                    manual_config=manual_config_tc, 
                    log_callback=log_callback, 
                    components=df_tc_clean['Component'], 
                    training_mode=training_mode
                )
                
                if results_tc:
                    best_perf = sorted(results_tc, key=lambda x: x['R2'], reverse=True)[0]
                    auto_winner = best_perf['Model']
                    print(f"Auto-Tune identified best Tc model: {auto_winner} (R2: {best_perf['R2']:.4f})")
                    
                    if target_tc == "Auto":
                        best_tc_model = auto_winner
                    else:
                        print(f"Using User Selected Model for Tc: {target_tc}")

            training_manager.check_interruption()

            # ----------------------------------------------------
            # 2. TRAIN PRODUCTION MODELS (Tune the Target)
            # ----------------------------------------------------
            imp_d33 = {}
            imp_tc = {}
            
            # Train D33
            if df_d33_clean is not None and len(df_d33_clean) > 10:
                training_manager.update_progress(60, f"Optimizing {best_d33_model} for d33...")
                
                X_d33 = create_feature_matrix(df_d33_clean['Component'])
                y_d33 = df_d33_clean['d33 (pC/N)']
                X_prod = X_d33[~y_d33.isna()]
                y_prod = y_d33[~y_d33.isna()]
                
                should_tune = True
                
                # Use specific params if available, else global params
                final_d33_params = d33_params if d33_params else params
                
                training_manager.check_interruption()
                
                _, imp_d33 = train_production_model(X_prod, y_prod, "d33", best_d33_model, final_d33_params, should_tune, training_mode, save_as_temp=True)
                print(f"Production d33 model ({best_d33_model}) trained (staged).")

            training_manager.check_interruption()

            # Train Tc
            if df_tc_clean is not None and len(df_tc_clean) > 10:
                training_manager.update_progress(80, f"Optimizing {best_tc_model} for Tc...")
                
                X_tc = create_feature_matrix(df_tc_clean['Component'])
                y_tc = df_tc_clean[tc_col]
                X_prod_tc = X_tc[~y_tc.isna()]
                y_prod_tc = y_tc[~y_tc.isna()]

                should_tune = True
                
                # Use specific params if available, else global params
                final_tc_params = tc_params if tc_params else params
                
                training_manager.check_interruption()

                _, imp_tc = train_production_model(X_prod_tc, y_prod_tc, "Tc", best_tc_model, final_tc_params, should_tune, training_mode, save_as_temp=True)
                print(f"Production Tc model ({best_tc_model}) trained (staged).")

            # Save Info (Used for Prediction Default)
            d33_mode_str = f"Auto ({best_d33_model})" if target_d33 == "Auto" else f"Manual ({best_d33_model})"
            tc_mode_str = f"Auto ({best_tc_model})" if target_tc == "Auto" else f"Manual ({best_tc_model})"
            
            save_active_model_info(best_d33_model, d33_mode_str, best_tc_model, tc_mode_str)
            
            # Ensure predictions are dictionaries (fallback to empty dict if None)
            save_insights(
                results_d33, 
                predictions_d33 if predictions_d33 else {}, 
                results_tc, 
                predictions_tc if predictions_tc else {}, 
                imp_d33, 
                imp_tc
            )
            
            # Finalize (Atomic Swap)
            finalize_staging_models()
            print("Model files finalized.")
            
            training_manager.complete_training()

        except InterruptedError as ie:
            # We must catch this specifically to log it as "Cancelled" not Failed
            # However, training_manager.abort_training sets state to idle or message.
            # But the loop might have raised this.
            training_manager.log("warning", "InterruptedError Caught in pipeline.")
        except Exception as e:
            training_manager.set_error(e, context="Training Pipeline")
        finally:
            pass


@router.get("/active-model")
async def get_active_model():
    return load_active_model_info()

@router.get("/export-report")
def export_report():
    from ml_engine import load_insights
    insights_data = load_insights()
    pdf_buffer = generate_report(insights_data)
    
    return StreamingResponse(
        pdf_buffer, 
        media_type="application/pdf", 
        headers={"Content-Disposition": "attachment; filename=piezo_ai_report.pdf"}
    )

@router.get("/insights")
def get_insights():
    from ml_engine import load_insights
    return load_insights()

@router.get("/dataset")
def get_dataset():
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    
    try:
        df = pd.read_csv(dataset_path)
        # Rename columns to match frontend expectations if necessary, or just return as is
        # Frontend expects: id, formula, d33, tc, source
        # CSV has: Component, d33 (pC/N), Tc (°C)
        
        # Create a list of dictionaries
        data = []
        for i, row in df.iterrows():
            data.append({
                "id": i + 1,
                "formula": row.get("Component", ""),
                "d33": row.get("d33 (pC/N)", 0),
                "tc": row.get("Tc (°C)", 0),
                "source": "Experimental" # Assuming all are experimental for now
            })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
