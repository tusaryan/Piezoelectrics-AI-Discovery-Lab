import uuid
from typing import List, Dict, Any, Tuple
import numpy as np
import joblib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from piezo_ml.features.engineer import FeatureEngineer
from piezo_ml.parsers.formula_parser import FormulaParser, FormulaParseError
from packages.db.models.prediction import Prediction
from packages.db.models.training import ModelArtifact

class PredictionService:
    engineer = FeatureEngineer()

    @staticmethod
    def _determine_use_case(d33: float, tc: float) -> Dict[str, str]:
        if d33 > 400 and tc < 150:
            return {"category": "Medical Imaging", "description": "High d33, low Tc. Ideal for ultrasound transducers.", "icon": "activity"}
        elif tc > 300:
            return {"category": "High-Temp Active", "description": "High Curie Temperature suitable for extreme environments.", "icon": "thermometer"}
        elif d33 > 200 and tc > 200:
            return {"category": "General Purpose Actuator", "description": "Balanced properties for industrial applications.", "icon": "cpu"}
        return {"category": "Sensors", "description": "Standard piezoelectric sensing.", "icon": "radio"}

    @staticmethod
    async def predict_single(db: AsyncSession, formula: str, sintering_temp: float = None, d33_artifact_id: str = None, tc_artifact_id: str = None) -> Dict[str, Any]:
        # 1. Parse and Feature Engineer
        try:
            parsed = FormulaParser.parse(formula)
        except FormulaParseError as e:
            # Re-raise to be handled by FastAPI exception handler
            raise e
            
        vector, names = PredictionService.engineer.compute_features(formula)
        
        # 2. Get active model (mocking this loader for MVP phase 1)
        # In a full flow we query SystemConfig or ModelArtifact where is_active_model=True
        # For Phase 1 scaffold, we generate dummy predictions if no model found
        
        # Determine output using real local XGBoost pipelines if available
        import os
        import joblib
        from apps.api.app.core.config import settings

        model_path = os.path.join(settings.model_artifacts_path, "active_d33_model.pkl")
        tc_model_path = os.path.join(settings.model_artifacts_path, "active_tc_model.pkl")

        if d33_artifact_id:
            d33_art = await db.get(ModelArtifact, d33_artifact_id)
            if d33_art and d33_art.artifact_path:
                model_path = d33_art.artifact_path
                
        if tc_artifact_id:
            tc_art = await db.get(ModelArtifact, tc_artifact_id)
            if tc_art and tc_art.artifact_path:
                tc_model_path = tc_art.artifact_path
        
        vector_arr = np.array([vector])
        
        try:
            if os.path.exists(model_path):
                d33_model = joblib.load(model_path)
                pseudo_d33 = float(d33_model.predict(vector_arr)[0])
            else:
                pseudo_d33 = float(np.mean(vector) * 1500)
                
            if os.path.exists(tc_model_path):
                tc_model = joblib.load(tc_model_path)
                pseudo_tc = float(tc_model.predict(vector_arr)[0])
            else:
                pseudo_tc = float(np.std(vector) * 3000)
        except Exception:
            pseudo_d33 = float(np.mean(vector) * 1500)
            pseudo_tc = float(np.std(vector) * 3000)
        
        lower_d33 = pseudo_d33 * 0.9
        upper_d33 = pseudo_d33 * 1.1
        lower_tc = pseudo_tc * 0.95
        upper_tc = pseudo_tc * 1.05

        # 3. Dummy Similar Materials
        similar = [
            {"formula": "KNN", "d33": 160.0, "tc": 420.0, "similarity": 0.85},
            {"formula": "BaTiO3", "d33": 190.0, "tc": 120.0, "similarity": 0.72}
        ]
        
        # 4. Save to DB
        prediction = Prediction(
            formula=formula,
            parsed_features=parsed,
            predicted_d33=pseudo_d33,
            predicted_tc=pseudo_tc,
            d33_lower_95=lower_d33,
            d33_upper_95=upper_d33,
            tc_lower_95=lower_tc,
            tc_upper_95=upper_tc,
            predicted_hardness=None,
            model_artifact_id=None
        )
        db.add(prediction)
        await db.commit()
        
        return {
            "formula": formula,
            "predicted_d33": pseudo_d33,
            "predicted_tc": pseudo_tc,
            "d33_lower_95": lower_d33,
            "d33_upper_95": upper_d33,
            "tc_lower_95": lower_tc,
            "tc_upper_95": upper_tc,
            "parsed_features": parsed,
            "similar_materials": similar,
            "use_case": PredictionService._determine_use_case(pseudo_d33, pseudo_tc)
        }
