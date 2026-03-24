import io
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

REPORTLAB_AVAILABLE = False
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    pass

class ReportGenerator:
    @staticmethod
    def generate_prediction_report(prediction_data: Dict[str, Any], config: Dict[str, Any] = None) -> bytes:
        buffer = io.BytesIO()
        
        if REPORTLAB_AVAILABLE:
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # --- Page 1: Cover ---
            c.setFont("Helvetica-Bold", 24)
            c.drawString(100, 700, "Piezo.AI Discovery Report")
            c.setFont("Helvetica", 14)
            c.drawString(100, 660, f"Prediction ID: {prediction_data.get('prediction_id', 'N/A')}")
            c.drawString(100, 630, f"Formula: {prediction_data.get('formula', 'Unknown')}")
            
            if 'predicted_d33' in prediction_data:
                c.drawString(100, 600, f"Predicted d33: {prediction_data['predicted_d33']} pC/N")
            if 'predicted_tc' in prediction_data:
                c.drawString(100, 570, f"Predicted Tc: {prediction_data['predicted_tc']} °C")
                
            c.showPage()
            
            # --- Page 2: Executive Summary ---
            if prediction_data.get('prediction_id') == 'dashboard-full':
                c.setFont("Helvetica-Bold", 18)
                c.drawString(100, 750, "Full System AI Insights & Discovery Report")
                c.setFont("Helvetica", 12)
                
                stats = prediction_data.get('system_stats', {})
                models = prediction_data.get('models', [])
                
                c.drawString(100, 710, f"Total Materials Analyzed: {stats.get('total_materials', 0)}")
                c.drawString(100, 680, f"Total Datasets Managed: {stats.get('total_datasets', 0)}")
                c.drawString(100, 650, f"Active AI Models Trained: {stats.get('total_models', 0)}")
                
                if models:
                    c.drawString(100, 600, "Top Model Performance:")
                    y_pos = 570
                    for m in models[:8]: # List top 8 models
                        model_name = m.get('model_name', 'Unknown Model')
                        target = m.get('target', 'Unknown')
                        r2 = m.get('r2_score')
                        r2_text = f"{r2:.4f}" if r2 is not None else "N/A"
                        c.drawString(120, y_pos, f"- {model_name} [{target.upper()}]: R2 = {r2_text}")
                        y_pos -= 25
                else:
                    c.drawString(100, 600, "No trained models found in the AI registry.")
                    c.drawString(100, 580, "Train a new Active Learning model to populate detailed SHAP graphs,")
                    c.drawString(100, 560, "Pareto Front optimizations, and AI-driven scientific insights here.")
            else:
                c.setFont("Helvetica-Bold", 18)
                c.drawString(100, 750, "Executive Summary & Model Details")
                c.setFont("Helvetica", 12)
                # For single predictions, use the active model registry
                active_model = prediction_data.get('active_model', 'RandomForest/XGBoost Ensemble')
                c.drawString(100, 710, f"Model Architecture: {active_model}")
                c.drawString(100, 680, "This report provides an automated synthesis of the AI predictions,")
                c.drawString(100, 650, "feature attributions, and certainty intervals for the analyzed material.")
            
            # Optional Pareto or SR sections based on config
            if config and config.get("include_pareto"):
                c.drawString(100, 500, "- Included: Pareto Optimization Fronts")
            if config and config.get("include_sr"):
                c.drawString(100, 470, "- Included: Symbolic Regression Discovered Equations")
            
            c.showPage()
            c.save()
            return buffer.getvalue()
        else:
            logger.warning("ReportLab not found. Generating dummy PDF byte stream.")
            # A minimal valid PDF byte sequence fallback
            buffer.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n")
            return buffer.getvalue()
