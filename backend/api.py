from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import io
import os
import json
from ml_engine import (
    create_feature_matrix, compare_models, train_production_model, 
    predict_properties, ALL_ELEMENTS, save_insights, save_active_model_info, load_active_model_info
)
from report_generator import generate_report

router = APIRouter()

# Global Training State
training_state = {
    "is_training": False,
    "progress": 0,
    "message": "Idle",
    "is_trained": False # Check if models exist on startup ideally, but for now default to False until trained or checked
}

# Check if models exist on startup
if os.path.exists("saved_models/d33_model.pkl") and os.path.exists("saved_models/Tc_model.pkl"):
    training_state["is_trained"] = True

class PredictRequest(BaseModel):
    formula: str

class TrainRequest(BaseModel):
    model_type: str = "Auto"
    params: Optional[Dict] = None

@router.get("/status")
def get_status():
    return training_state

@router.post("/predict")
def predict(request: PredictRequest):
    if training_state["is_training"]:
        raise HTTPException(status_code=503, detail="Model is currently retraining. Please wait.")
    
    if not training_state["is_trained"]:
         # Double check file existence just in case
        if not (os.path.exists("saved_models/d33_model.pkl") and os.path.exists("saved_models/Tc_model.pkl")):
             raise HTTPException(status_code=400, detail="Models not trained yet. Please go to the Retraining tab.")
        else:
             training_state["is_trained"] = True

    d33, tc, composition = predict_properties(request.formula)
    if d33 is None:
        raise HTTPException(status_code=400, detail="Prediction failed.")
    
    # Filter out zero values for cleaner display
    active_composition = {k: v for k, v in composition.items() if v > 0}
    
    return {"d33": d33, "Tc": tc, "composition": active_composition}

@router.post("/train")
async def train(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    model_type: str = "Auto",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    c_param: float = 100.0,
    epsilon: float = 0.1
):
    if training_state["is_training"]:
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
        
    # Construct params dict based on model_type
    params = {}
    if model_type != "Auto":
        if model_type in ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]:
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
        
        if model_type in ["XGBoost", "LightGBM", "Gradient Boosting"]:
            params["learning_rate"] = learning_rate
            
        if model_type == "SVM (SVR)":
            params["C"] = c_param
            params["epsilon"] = epsilon

    # Trigger training in background
    background_tasks.add_task(run_training_pipeline, df, model_type, params)
    
    return {"message": "Training started."}

def run_training_pipeline(df, model_type, params):
    global training_state
    training_state["is_training"] = True
    training_state["progress"] = 0
    training_state["message"] = "Initializing training..."
    
    try:
        # Process d33
        training_state["progress"] = 10
        training_state["message"] = "Training d33 model..."
        if 'd33 (pC/N)' in df.columns:
            df_d33 = df.dropna(subset=['Component', 'd33 (pC/N)'])
            X_d33 = create_feature_matrix(df_d33['Component'])
            y_d33 = df_d33['d33 (pC/N)']
            train_production_model(X_d33, y_d33, "d33", model_type, params)
            
        # Process Tc
        training_state["progress"] = 40
        training_state["message"] = "Training Tc model..."
        tc_col = next((c for c in df.columns if 'Tc' in c), None)
        if tc_col:
            df_tc = df.dropna(subset=['Component', tc_col])
            X_tc = create_feature_matrix(df_tc['Component'])
            y_tc = df_tc[tc_col]
            train_production_model(X_tc, y_tc, "Tc", model_type, params)

        # Run model comparison and save insights
        training_state["progress"] = 60
        training_state["message"] = "Comparing d33 algorithms..."
        results_d33, predictions_d33 = None, None
        results_tc, predictions_tc = None, None

        manual_config = {"model_type": model_type, "params": params} if model_type != "Auto" else None

        if 'd33 (pC/N)' in df.columns:
            df_d33 = df.dropna(subset=['Component', 'd33 (pC/N)'])
            X = create_feature_matrix(df_d33['Component'])
            y = df_d33['d33 (pC/N)']
            results_d33, predictions_d33 = compare_models(X, y, "d33", manual_config)

        training_state["progress"] = 80
        training_state["message"] = "Comparing Tc algorithms..."
        tc_col = next((c for c in df.columns if 'Tc' in c), None)
        if tc_col:
            df_tc = df.dropna(subset=['Component', tc_col])
            X = create_feature_matrix(df_tc['Component'])
            y = df_tc[tc_col]
            results_tc, predictions_tc = compare_models(X, y, "Tc", manual_config)
            
        # Determine active model info for saving
        d33_model_name = "XGBoost (Auto)"
        d33_mode = "Auto"
        tc_model_name = "XGBoost (Auto)"
        tc_mode = "Auto"

        if manual_config:
            d33_model_name = manual_config.get("model_type", "Custom")
            d33_mode = "Manual"
            tc_model_name = manual_config.get("model_type", "Custom")
            tc_mode = "Manual"
        
        save_active_model_info(d33_model_name, d33_mode, tc_model_name, tc_mode)
        
        save_insights(results_d33, predictions_d33, results_tc, predictions_tc)
        
        training_state["progress"] = 100
        training_state["message"] = "Training complete!"
        training_state["is_trained"] = True

    except Exception as e:
        training_state["message"] = f"Error: {str(e)}"
    finally:
        training_state["is_training"] = False

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
