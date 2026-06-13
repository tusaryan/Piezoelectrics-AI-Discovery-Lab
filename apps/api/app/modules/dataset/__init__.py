"""
Piezo.AI — Dataset Module
============================
Upload, map, review, and explore CSV datasets.

NOTE: This module is a DUMB PIPE — all ML logic (formula parsing,
feature engineering) is delegated to packages/ml-core.
"""

from app.modules.dataset.router import router

__all__ = ["router"]
