"""
Piezo.AI — API Modules
========================
Each sub-package represents one section of the app:
- dataset/   → Upload, map, review, explore CSV datasets
- training/  → Configure and execute ML training pipelines
- prediction/ → Unified prediction (bulk + composite + hardness)
- optimization/ → Crystal structure analysis + NSGA-II optimization
- interpret/ → SHAP analysis + Symbolic Regression (PySR)
- settings/  → Models library, system env, API config
- dashboard/ → Stats, reports, model management

NOTE: These modules are DUMB PIPES — they call into packages/ml-core
for all ML computations. Zero ML logic in this directory.
"""
