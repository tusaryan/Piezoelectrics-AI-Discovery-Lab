"""
Piezo.AI ML Core — Reporting Subpackage
==========================================
Premium PDF report generation with embedded charts.
"""

from piezo_ml.reporting.report_builder import PiezoReportBuilder
from piezo_ml.reporting.chart_generator import (
    generate_r2_rmse_chart,
    generate_target_distribution_chart,
    generate_model_performance_chart,
    generate_convergence_chart,
    generate_usage_chart,
)

__all__ = [
    "PiezoReportBuilder",
    "generate_r2_rmse_chart",
    "generate_target_distribution_chart",
    "generate_model_performance_chart",
    "generate_convergence_chart",
    "generate_usage_chart",
]
