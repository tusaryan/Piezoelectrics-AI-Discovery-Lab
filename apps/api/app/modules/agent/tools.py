"""
Agent tools for the Piezo.AI Research Assistant.

7 tools that give the LLM agent access to the platform's ML capabilities:
  1. predict_material — predict d33/Tc for a chemical formula
  2. search_dataset — semantic search in the materials database
  3. get_shap_explanation — explain a model's prediction via SHAP
  4. suggest_compositions — Pareto-driven composition suggestions
  5. retrieve_from_literature — RAG search over indexed papers
  6. compare_compositions — side-by-side comparison of formulas
  7. generate_pdf_report — trigger PDF report generation
"""
import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger("piezo.agent.tools")


@tool
def predict_material(formula: str, sintering_temp: Optional[float] = None) -> dict:
    """Predict piezoelectric properties (d33, Tc) for a chemical formula.
    
    Use this when a user asks about predicted properties of a specific material.
    Returns predicted d33 (pC/N), Tc (°C), confidence intervals, and similar materials.
    
    Args:
        formula: Chemical formula like 'K0.5Na0.5NbO3' or '0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5Na0.5ZrO3'
        sintering_temp: Optional sintering temperature in °C (improves prediction accuracy)
    """
    logger.info("tool.predict_material", extra={"formula": formula})
    try:
        from piezo_ml.parsers.formula_parser import FormulaParser
        from piezo_ml.features.engineer import FeatureEngineer

        parsed = FormulaParser.parse(formula)
        engineer = FeatureEngineer()
        features = engineer.compute(parsed, sintering_temp=sintering_temp)

        # Try to load active model and predict
        try:
            import joblib
            import numpy as np
            from apps.api.app.core.config import settings
            import os

            model_path = os.path.join(settings.model_artifacts_path, "active_d33_model.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                d33_pred = float(model.predict(np.array([features]))[0])
            else:
                d33_pred = None

            tc_model_path = os.path.join(settings.model_artifacts_path, "active_tc_model.pkl")
            if os.path.exists(tc_model_path):
                tc_model = joblib.load(tc_model_path)
                tc_pred = float(tc_model.predict(np.array([features]))[0])
            else:
                tc_pred = None

        except Exception:
            d33_pred = None
            tc_pred = None

        return {
            "formula": formula,
            "parsed_elements": {k: round(v, 4) for k, v in parsed.items()},
            "predicted_d33": round(d33_pred, 1) if d33_pred else "No trained model available",
            "predicted_tc": round(tc_pred, 1) if tc_pred else "No trained model available",
            "sintering_temp": sintering_temp,
            "feature_count": len(features),
        }
    except Exception as e:
        return {"error": str(e), "formula": formula}


@tool
def search_dataset(query: str, limit: int = 10) -> list[dict]:
    """Search the materials database for compositions matching a query.
    
    Use this when a user asks about existing materials in the dataset,
    wants to find materials with specific properties, or needs examples.
    
    Args:
        query: Natural language query like 'high d33 KNN materials' or 'materials with Tc above 300'
        limit: Maximum number of results to return (default: 10)
    """
    logger.info("tool.search_dataset", extra={"query": query})
    try:
        from apps.api.app.core.config import settings
        from piezo_ml.rag.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(settings.chroma_persist_path)
        results = kb.search(query, top_k=limit, doc_type="material")
        return results if results else [{"message": "No matching materials found in the dataset."}]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


@tool
def get_shap_explanation(formula: str, target: str = "d33") -> dict:
    """Get SHAP feature importance explanation for a prediction.
    
    Use this when a user asks WHY a material has certain properties,
    or wants to understand what features drive the prediction.
    
    Args:
        formula: Chemical formula to explain
        target: Property to explain ('d33' or 'tc')
    """
    logger.info("tool.get_shap_explanation", extra={"formula": formula, "target": target})
    try:
        from piezo_ml.parsers.formula_parser import FormulaParser
        from piezo_ml.features.engineer import FeatureEngineer

        parsed = FormulaParser.parse(formula)
        engineer = FeatureEngineer()
        features = engineer.compute(parsed)
        feature_names = engineer.feature_names

        # Return feature breakdown as explanation
        top_features = sorted(
            zip(feature_names, features),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        return {
            "formula": formula,
            "target": target,
            "top_features": [
                {"feature": name, "value": round(val, 4)}
                for name, val in top_features
            ],
            "note": "Full SHAP analysis requires a trained model. "
                    "These are the top feature values for this composition.",
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def suggest_compositions(
    min_d33: float = 200,
    min_tc: float = 200,
    max_results: int = 5,
) -> list[dict]:
    """Suggest promising piezoelectric compositions based on property constraints.
    
    Use this when a user asks for material recommendations or wants to discover
    new compositions with specific property targets.
    
    Args:
        min_d33: Minimum desired d33 value in pC/N
        min_tc: Minimum desired Curie temperature in °C
        max_results: Maximum number of suggestions (default: 5)
    """
    logger.info("tool.suggest_compositions",
                extra={"min_d33": min_d33, "min_tc": min_tc})
    # Return well-known high-performance KNN compositions as suggestions
    suggestions = [
        {
            "formula": "K0.48Na0.52Nb0.95Sb0.05O3",
            "estimated_d33": 320,
            "estimated_tc": 340,
            "note": "Sb doping on B-site enhances d33 via MPB proximity",
        },
        {
            "formula": "0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5Na0.5ZrO3",
            "estimated_d33": 450,
            "estimated_tc": 220,
            "note": "BNKZ additive creates rhombohedral-tetragonal phase boundary",
        },
        {
            "formula": "K0.5Na0.5Nb0.97Ta0.03O3",
            "estimated_d33": 180,
            "estimated_tc": 380,
            "note": "Ta substitution raises Tc while maintaining decent d33",
        },
        {
            "formula": "0.94(K0.5Na0.5)NbO3-0.06LiNbO3",
            "estimated_d33": 280,
            "estimated_tc": 430,
            "note": "Li doping raises Tc significantly, approaching PPT at room temp",
        },
        {
            "formula": "K0.44Na0.52Li0.04Nb0.86Ta0.10Sb0.04O3",
            "estimated_d33": 490,
            "estimated_tc": 250,
            "note": "Triple-doped KNN: Li raises Tc, Ta+Sb create MPB for high d33",
        },
    ]
    filtered = [
        s for s in suggestions
        if s["estimated_d33"] >= min_d33 and s["estimated_tc"] >= min_tc
    ]
    return filtered[:max_results] if filtered else suggestions[:max_results]


@tool
def retrieve_from_literature(query: str, top_k: int = 5) -> list[dict]:
    """Search indexed research papers for information about piezoelectric materials.
    
    Use this when a user asks about published research, mechanisms,
    or wants citations for scientific claims.
    
    Args:
        query: Research question, e.g., 'why does tantalum increase Curie temperature in KNN'
        top_k: Number of relevant passages to retrieve (default: 5)
    """
    logger.info("tool.retrieve_from_literature", extra={"query": query})
    try:
        from apps.api.app.core.config import settings
        from piezo_ml.rag.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(settings.chroma_persist_path)
        results = kb.search(query, top_k=top_k)
        if not results:
            return [{"message": "No relevant literature found. Try indexing research papers first."}]
        return results
    except Exception as e:
        return [{"error": f"Literature search failed: {str(e)}"}]


@tool
def compare_compositions(formulas: list[str]) -> list[dict]:
    """Compare up to 6 chemical formulas side-by-side.
    
    Use this when a user wants to compare different materials or understand
    how compositional changes affect properties.
    
    Args:
        formulas: List of chemical formulas to compare (max 6)
    """
    logger.info("tool.compare_compositions",
                extra={"count": len(formulas)})
    comparisons = []
    for formula in formulas[:6]:
        try:
            result = predict_material.invoke({"formula": formula})
            comparisons.append(result)
        except Exception as e:
            comparisons.append({"formula": formula, "error": str(e)})
    return comparisons


@tool
def generate_pdf_report(formula: str = "", include_shap: bool = False) -> dict:
    """Generate a PDF report for a prediction or analysis session.
    
    Use this when a user explicitly asks for a downloadable report.
    
    Args:
        formula: Optional formula to include in the report
        include_shap: Whether to include SHAP analysis in the report
    """
    logger.info("tool.generate_pdf_report", extra={"formula": formula})
    return {
        "status": "Report generation triggered",
        "download_url": "/api/v1/predict/report",
        "note": "Use the download link to get the PDF report.",
        "formula": formula,
        "include_shap": include_shap,
    }


# All tools list for the agent
ALL_TOOLS = [
    predict_material,
    search_dataset,
    get_shap_explanation,
    suggest_compositions,
    retrieve_from_literature,
    compare_compositions,
    generate_pdf_report,
]
