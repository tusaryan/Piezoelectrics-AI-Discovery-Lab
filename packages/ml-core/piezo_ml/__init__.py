"""
Piezo.AI ML Core
==================
ALL machine learning logic lives here. No exceptions.

Subpackages:
- registry/    → Central Element Registry (single source of truth for 33+ elements)
- parsers/     → Formula parsing, validation, normalization
- features/    → Feature engineering (elemental fractions + physics descriptors)
- pipeline/    → Data loading, cleaning, validation, training orchestration
- models/      → Inference engine, model registry, composite predictor, use-case mapper
- evaluation/  → SHAP analyzer, metrics calculation
- optimization/ → NSGA-II Pareto front, crystal structure analysis
- symbolic_regression/ → PySR integration
- reporting/   → PDF report generation (ReportLab)
"""

__version__ = "2.1.0"
