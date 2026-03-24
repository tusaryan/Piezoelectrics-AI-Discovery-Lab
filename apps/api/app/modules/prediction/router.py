import io
import csv
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.app.core.database import get_db
from apps.api.app.modules.prediction.schemas import (
    SinglePredictRequest, SinglePredictResponse,
    ComparePredictRequest, ComparePredictResponse
)
from apps.api.app.modules.prediction.service import PredictionService
from piezo_ml.features.engineer import UnsupportedElementError
from piezo_ml.parsers.formula_parser import FormulaParseError
from fastapi import HTTPException

router = APIRouter(prefix="/api/v1/predict", tags=["prediction"])

@router.post("/single", response_model=SinglePredictResponse)
async def predict_single(request: SinglePredictRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await PredictionService.predict_single(
            db, 
            request.formula, 
            request.sintering_temp,
            request.d33_artifact_id,
            request.tc_artifact_id
        )
        return {
            "data": result,
            "meta": {}
        }
    except FormulaParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UnsupportedElementError as e:
        elements_str = ", ".join(e.elements)
        raise HTTPException(
            status_code=400, 
            detail=f"Whoops! We detected unsupported elements ({elements_str}) in your formula. We're currently working enthusiastically to expand our AI model's capabilities to support more of the periodic table! For now, please stick to our 25 supported elements."
        )

@router.post("/compare", response_model=ComparePredictResponse)
async def predict_compare(request: ComparePredictRequest, db: AsyncSession = Depends(get_db)):
    results = []
    for formula in request.formulas[:6]:  # Limit to 6 to prevent abuse
        try:
            res = await PredictionService.predict_single(db, formula)
            results.append(res)
        except Exception:
            pass # Skip invalid formulas in comparison mode
            
    return {
        "data": results,
        "meta": {"count": len(results)}
    }

@router.post("/report")
async def generate_report(request: dict, db: AsyncSession = Depends(get_db)):
    from piezo_ml.reporting.pdf_generator import ReportGenerator
    
    prediction_id = request.get("prediction_id", "N/A")
    
    prediction_data = {
        "prediction_id": prediction_id,
        "formula": request.get("formula", "Multiple (Dashboard)" if prediction_id == "dashboard-full" else "Unknown")
    }
    
    if prediction_id == "dashboard-full":
        # Import services here to avoid circular dependencies if any
        from apps.api.app.modules.system.service import SystemService
        from apps.api.app.modules.training.service import ModelService
        
        # Query real DB stats for the dashboard PDF
        stats = await SystemService.get_stats(db)
        models = await ModelService.list_models(db)
        
        prediction_data["system_stats"] = {
            "total_materials": stats.get("total_materials", 0),
            "total_datasets": stats.get("total_datasets", 0),
            "total_models": stats.get("total_models", 0),
            "total_training_jobs": stats.get("total_training_jobs", 0),
        }
        prediction_data["models"] = [
            {"model_name": m.model_name, "target": m.target, "r2_score": m.r2_test} 
            for m in models
        ]
    config = {
        "include_pareto": request.get("include_pareto", False),
        "include_sr": request.get("include_sr", False)
    }
    
    pdf_bytes = ReportGenerator.generate_prediction_report(prediction_data, config)
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=piezo_report.pdf"}
    )

@router.post("/batch")
async def predict_batch(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    contents = await file.read()
    decoded = contents.decode('utf-8')
    csv_reader = csv.DictReader(io.StringIO(decoded))
    
    results = []
    for num, row in enumerate(csv_reader):
        if num > 1000: # hard limit for MVP
            break
            
        formula = row.get("formula", "")
        if not formula:
            continue
            
        try:
            temp = float(row.get("sintering_temp")) if row.get("sintering_temp") else None
            res = await PredictionService.predict_single(db, formula, temp)
            results.append({
                "formula": formula,
                "predicted_d33": res["predicted_d33"],
                "predicted_tc": res["predicted_tc"],
                "d33_lower_95": res["d33_lower_95"],
                "d33_upper_95": res["d33_upper_95"],
                "tc_lower_95": res["tc_lower_95"],
                "tc_upper_95": res["tc_upper_95"]
            })
        except Exception as e:
            results.append({
                "formula": formula,
                "error": str(e)
            })

    output = io.StringIO()
    if results:
        fieldnames = ["formula", "predicted_d33", "predicted_tc", "d33_lower_95", "d33_upper_95", "tc_lower_95", "tc_upper_95", "error"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
    )
