from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import io
import os
from ml_engine import (
    create_feature_matrix, compare_models, train_production_model, 
    predict_properties, ALL_ELEMENTS, save_insights
)

router = APIRouter()

class PredictRequest(BaseModel):
    formula: str

class TrainRequest(BaseModel):
    model_type: str = "Auto"
    params: Optional[Dict] = None

@router.post("/predict")
def predict(request: PredictRequest):
    d33, tc, composition = predict_properties(request.formula)
    if d33 is None:
        raise HTTPException(status_code=400, detail="Models not trained yet.")
    
    # Filter out zero values for cleaner display
    active_composition = {k: v for k, v in composition.items() if v > 0}
    
    return {"d33": d33, "Tc": tc, "composition": active_composition}

@router.post("/train")
async def train(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    model_type: str = "Auto",
    n_estimators: int = 100,
    learning_rate: float = 0.1
):
    # Read dataset
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Check columns
    if 'Component' not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset must have 'Component' column.")
        
    # Trigger training in background
    background_tasks.add_task(run_training_pipeline, df, model_type, {"n_estimators": n_estimators, "learning_rate": learning_rate})
    
    return {"message": "Training started in background."}

def run_training_pipeline(df, model_type, params):
    # Process d33
    if 'd33 (pC/N)' in df.columns:
        df_d33 = df.dropna(subset=['Component', 'd33 (pC/N)'])
        X_d33 = create_feature_matrix(df_d33['Component'])
        y_d33 = df_d33['d33 (pC/N)']
        train_production_model(X_d33, y_d33, "d33", model_type, params)
        
    # Process Tc
    tc_col = next((c for c in df.columns if 'Tc' in c), None)
    if tc_col:
        df_tc = df.dropna(subset=['Component', tc_col])
        X_tc = create_feature_matrix(df_tc['Component'])
        y_tc = df_tc[tc_col]
        train_production_model(X_tc, y_tc, "Tc", model_type, params)

    # Run model comparison and save insights
    results_d33, predictions_d33 = None, None
    results_tc, predictions_tc = None, None

    if 'd33 (pC/N)' in df.columns:
        df_d33 = df.dropna(subset=['Component', 'd33 (pC/N)'])
        X = create_feature_matrix(df_d33['Component'])
        y = df_d33['d33 (pC/N)']
        results_d33, predictions_d33 = compare_models(X, y, "d33")

    tc_col = next((c for c in df.columns if 'Tc' in c), None)
    if tc_col:
        df_tc = df.dropna(subset=['Component', tc_col])
        X = create_feature_matrix(df_tc['Component'])
        y = df_tc[tc_col]
        results_tc, predictions_tc = compare_models(X, y, "Tc")
        
    save_insights(results_d33, predictions_d33, results_tc, predictions_tc)

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
