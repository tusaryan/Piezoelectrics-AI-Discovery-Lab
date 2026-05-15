"""
Piezo.AI — Settings Router
============================
REST endpoints for system settings, config, LLM, elements, danger zone.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.settings import schemas, service

logger = logging.getLogger(__name__)
router = APIRouter()


# ── System Environment ──────────────────────

@router.get("/system", response_model=schemas.SystemEnvironment)
async def get_system_environment(db: AsyncSession = Depends(get_db)):
    """Get system overview stats."""
    data = await service.get_system_environment(db)
    return schemas.SystemEnvironment(**data)


# ── App Configuration ──────────────────────

@router.get("/config")
async def get_app_config():
    """Get current .env config values (safe subset)."""
    return service.get_app_config()


@router.put("/config")
async def update_app_config(body: schemas.AppConfigUpdate):
    """Update .env variables."""
    return service.update_app_config(body.updates)


@router.post("/config/logo")
async def upload_logo(file: UploadFile = File(...)):
    """Upload a logo image to /public."""
    filename = file.filename or "logo.png"
    content = await file.read()
    if len(content) > 2_000_000:  # 2MB limit
        raise HTTPException(status_code=400, detail="Logo file too large. Max 2MB.")
    result = service.upload_logo(filename, content)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/config/import", response_model=schemas.EnvImportResponse)
async def import_env_file(file: UploadFile = File(...)):
    """Import .env file to update configuration."""
    # Validate file type
    filename = file.filename or ""
    valid_names = {".env", ".env.local", ".env.example", ".env.production",
                   ".env.development", ".env.staging", ".env.test"}
    valid_ext = filename.startswith(".env") or filename in valid_names
    if not valid_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{filename}'. Only .env and .env.* files "
                   f"(e.g., .env, .env.local, .env.example) are accepted."
        )
    # Validate content type
    content = await file.read()
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File is not valid UTF-8 text. Only plain text .env files are accepted."
        )
    if len(text_content) > 100_000:
        raise HTTPException(
            status_code=400,
            detail="File too large. .env files should be under 100KB."
        )
    return service.import_env_file(text_content)


# ── LLM Configuration ──────────────────────

@router.get("/llm/status", response_model=schemas.LlmConfigResponse)
async def get_llm_status():
    """Get current LLM configuration and status."""
    return service.get_llm_config()


@router.put("/llm", response_model=schemas.LlmConfigResponse)
async def update_llm_config(body: schemas.LlmConfigUpdate):
    """Update LLM settings."""
    data = body.model_dump(exclude_none=True)
    return service.update_llm_config(data)


@router.get("/llm/providers")
async def get_llm_providers():
    """Get list of supported LLM providers."""
    return service.get_llm_providers()


# ── Element Registry ──────────────────────

@router.get("/elements", response_model=schemas.ElementRegistryResponse)
async def get_element_registry():
    """Get full element registry state."""
    return service.get_element_registry()


@router.post("/elements/pending")
async def add_pending_element(body: schemas.PendingElementRequest):
    """Add element to pending list."""
    result = service.add_pending_element(body.symbol, body.categories)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/elements/pending/{symbol}")
async def remove_pending_element(symbol: str):
    """Remove element from pending list."""
    result = service.remove_pending_element(symbol)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.delete("/elements/supported/{symbol}")
async def remove_supported_element(symbol: str):
    """Remove a user-added element from the supported registry."""
    result = service.remove_supported_element(symbol)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/elements/bootstrap", response_model=schemas.BootstrapResult)
async def bootstrap_elements():
    """Trigger bootstrap for pending elements."""
    return service.bootstrap_elements()


# ── Custom Properties ──────────────────────

@router.post("/properties")
async def add_custom_property(body: schemas.AddPropertyRequest):
    """Add a custom property key to the element registry."""
    result = service.add_custom_property(body.property_key)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/properties/{property_key}")
async def remove_custom_property(property_key: str):
    """Remove a user-added custom property."""
    result = service.remove_custom_property(property_key)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Field Schema Manager ──────────────────────

@router.get("/fields")
async def get_field_schema():
    """Get complete field schema (all fields, types, categories, aliases)."""
    return service.get_field_schema()


@router.post("/fields")
async def add_user_field(body: schemas.AddFieldRequest):
    """Add a new user-defined field to the schema."""
    result = service.add_user_field(body.model_dump())
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# Static routes MUST come before /{field_name} to avoid path param conflicts
@router.get("/fields/export")
async def export_field_schema():
    """Export the full field schema as portable JSON."""
    return service.export_field_schema()


@router.post("/fields/import")
async def import_field_schema(body: schemas.FieldSchemaImportRequest):
    """Import field customizations from exported schema."""
    result = service.import_field_schema(body.schema_data)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/fields/{field_name}")
async def remove_user_field(field_name: str):
    """Remove a user-added field from the schema."""
    result = service.remove_user_field(field_name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/fields/{field_name}/categories")
async def add_field_category(field_name: str, body: schemas.AddCategoryRequest):
    """Add a category value to an existing categorical field."""
    result = service.add_field_category(field_name, body.value)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/fields/{field_name}/categories/{value}")
async def remove_field_category(field_name: str, value: str):
    """Remove a user-added category value from a field."""
    result = service.remove_field_category(field_name, value)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/fields/{field_name}/aliases")
async def add_field_alias(field_name: str, body: schemas.AddAliasRequest):
    """Add an alias mapping for a field."""
    result = service.add_field_alias(field_name, body.alias, body.canonical)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Reset ──────────────────────

@router.post("/reset/elements", response_model=schemas.ResetResponse)
async def reset_elements_and_properties(body: schemas.DangerActionRequest):
    """Reset elements and properties to defaults."""
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")
    return service.reset_elements_and_properties()


@router.post("/reset/all", response_model=schemas.ResetResponse)
async def reset_all_settings(body: schemas.DangerActionRequest):
    """Factory reset: restore all settings to defaults."""
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")
    return service.reset_all_settings()


# ── Model Library ──────────────────────

@router.get("/models")
async def get_models(db: AsyncSession = Depends(get_db)):
    """Get all trained models for settings library."""
    from piezo_db.models import TrainedModel
    from sqlalchemy import select

    result = await db.execute(
        select(TrainedModel).order_by(TrainedModel.created_at.desc())
    )
    models = result.scalars().all()
    return [
        schemas.SettingsModelInfo(
            id=str(m.id),
            display_name=m.display_name,
            target=m.target,
            algorithm=m.algorithm,
            r2_score=m.r2_score,
            rmse=m.rmse,
            feature_dim=m.feature_dim,
            n_train_samples=m.n_train_samples,
            n_test_samples=m.n_test_samples,
            is_default=m.is_default,
            model_file_path=m.model_file_path,
            created_at=str(m.created_at),
        )
        for m in models
    ]


@router.patch("/models/{model_id}/rename")
async def rename_model(
    model_id: UUID, body: schemas.ModelRenameRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rename a trained model (UUID stays same)."""
    from piezo_db.models import TrainedModel

    model = await db.get(TrainedModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.display_name = body.new_name
    await db.commit()
    return {"id": str(model.id), "display_name": model.display_name}


@router.patch("/models/{model_id}/default")
async def set_default_model(
    model_id: UUID, db: AsyncSession = Depends(get_db),
):
    """Set model as default for its target."""
    from piezo_db.models import TrainedModel
    from sqlalchemy import select, update

    model = await db.get(TrainedModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    await db.execute(
        update(TrainedModel)
        .where(TrainedModel.target == model.target, TrainedModel.is_default == True)
        .values(is_default=False)
    )
    model.is_default = True
    await db.commit()
    return {"id": str(model.id), "is_default": True, "target": model.target}


@router.delete("/models/{model_id}")
async def delete_model(model_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete a trained model."""
    from piezo_db.models import TrainedModel

    model = await db.get(TrainedModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = service.PROJECT_ROOT / model.model_file_path.lstrip("./")
    if model_path.exists():
        model_path.unlink()

    await db.delete(model)
    await db.commit()
    return {"success": True, "message": f"Deleted model {model.display_name}"}


@router.post("/models/batch-delete", response_model=schemas.DangerActionResponse)
async def batch_delete_models(body: schemas.BatchDeleteRequest, db: AsyncSession = Depends(get_db)):
    """Delete multiple models at once."""
    result = await service.batch_delete_models(db, body.model_ids)
    return result


# ── Danger Zone ──────────────────────

@router.post("/danger/purge-models", response_model=schemas.DangerActionResponse)
async def purge_all_models(
    body: schemas.DangerActionRequest, db: AsyncSession = Depends(get_db),
):
    """Purge ALL trained models from DB and filesystem."""
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")
    return await service.purge_all_models(db)


@router.post("/danger/clear-cache", response_model=schemas.DangerActionResponse)
async def clear_prediction_cache(
    body: schemas.DangerActionRequest, db: AsyncSession = Depends(get_db),
):
    """Clear all predictions and batch data."""
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")
    return await service.clear_prediction_cache(db)


# ── GNN Status ──────────────────────

@router.get("/gnn/status", response_model=schemas.GnnStatusResponse)
async def get_gnn_status():
    """Check GNN/CHGNet dependency status."""
    return service.get_gnn_status()
