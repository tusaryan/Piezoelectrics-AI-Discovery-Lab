"""
Piezo.AI Database Package
===========================
SQLAlchemy models and Alembic migrations for the Piezo.AI database.

Tables defined here (see 02-cross-cutting-and-build-plan.md §5.6 for full schema):
- datasets          → Uploaded CSV metadata
- materials         → Individual dataset rows (post-column-mapping)
- training_jobs     → Training pipeline execution state
- trained_models    → Trained model metadata (UUID-stable)
- predictions       → Prediction history (single + batch rows)
- prediction_batches → Batch prediction job metadata
"""
