"""
Piezo.AI — Settings Service
=============================
Business logic for settings endpoints.
All DB operations + .env file management + element registry access.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)

# Project root (apps/api/app/modules/settings/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parents[5]
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

# Custom additions tracking file — persists user-added elements/properties
CUSTOM_ADDITIONS_PATH = PROJECT_ROOT / "resources" / ".settings-customizations.json"

# Default .env backup for reset
DEFAULT_ENV_BACKUP_PATH = PROJECT_ROOT / "resources" / ".env.defaults"


def _load_custom_additions() -> dict[str, Any]:
    """Load user customizations tracking file."""
    if not CUSTOM_ADDITIONS_PATH.exists():
        return {"added_elements": [], "added_properties": []}
    try:
        with open(CUSTOM_ADDITIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"added_elements": [], "added_properties": []}


def _save_custom_additions(data: dict[str, Any]) -> None:
    """Save user customizations tracking file."""
    CUSTOM_ADDITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_ADDITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── .env File Helpers ──────────────────────

def _read_env_file(path: Path | None = None) -> dict[str, str]:
    """Read an .env file into dict."""
    target = path or ENV_PATH
    env_vars: dict[str, str] = {}
    if not target.exists():
        return env_vars
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip()
    return env_vars


def _write_env_file(updates: dict[str, str]) -> None:
    """Update .env file, preserving comments and structure."""
    if not ENV_PATH.exists():
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            for key, value in updates.items():
                f.write(f"{key}={value}\n")
        return

    lines: list[str] = []
    updated_keys: set[str] = set()

    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    lines.append(f"{key}={updates[key]}\n")
                    updated_keys.add(key)
                    continue
            lines.append(line)

    for key, value in updates.items():
        if key not in updated_keys:
            lines.append(f"{key}={value}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _backup_default_env() -> None:
    """Create a backup of the pristine .env for reset capability."""
    if not DEFAULT_ENV_BACKUP_PATH.exists() and ENV_PATH.exists():
        DEFAULT_ENV_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ENV_PATH, DEFAULT_ENV_BACKUP_PATH)


# On module load, save a backup if none exists
_backup_default_env()

# Public directory for static files (logo etc.)
WEB_PUBLIC_DIR = PROJECT_ROOT / "apps" / "web" / "public"


def upload_logo(filename: str, content: bytes) -> dict[str, Any]:
    """Save uploaded logo to /public and update .env path."""
    import hashlib
    import time

    # Generate unique name to avoid conflicts
    ext = Path(filename).suffix.lower() or ".png"
    allowed_ext = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".ico", ".gif"}
    if ext not in allowed_ext:
        return {"error": f"Unsupported image format '{ext}'. Allowed: {', '.join(sorted(allowed_ext))}"}

    # Create unique filename
    hash_suffix = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:8]
    safe_name = f"app-logo-{hash_suffix}{ext}"

    # Save to public dir
    WEB_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEB_PUBLIC_DIR / safe_name
    with open(dest, "wb") as f:
        f.write(content)

    # Update .env
    logo_path = f"/{safe_name}"
    _write_env_file({
        "APP_LOGO_PATH": logo_path,
        "NEXT_PUBLIC_APP_LOGO_PATH": logo_path,
    })

    return {
        "message": f"Logo uploaded and saved as {safe_name}",
        "path": logo_path,
    }


# ── System Environment ──────────────────────

async def get_system_environment(db: AsyncSession) -> dict[str, Any]:
    """Get system overview stats."""
    from piezo_db.models import Dataset, TrainedModel, Prediction

    ds_count = (await db.execute(select(func.count(Dataset.id)))).scalar() or 0
    total_rows_result = (await db.execute(
        select(func.coalesce(func.sum(Dataset.total_rows), 0))
    )).scalar()
    total_rows = int(total_rows_result) if total_rows_result else 0
    model_count = (await db.execute(select(func.count(TrainedModel.id)))).scalar() or 0
    pred_count = (await db.execute(select(func.count(Prediction.id)))).scalar() or 0

    db_size_mb = 0.0
    try:
        result = await db.execute(
            text("SELECT pg_database_size(current_database()) / 1048576.0")
        )
        db_size_mb = round(float(result.scalar() or 0), 2)
    except Exception:
        pass

    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "dataset_count": ds_count,
        "total_rows": total_rows,
        "trained_model_count": model_count,
        "prediction_count": pred_count,
        "db_size_mb": db_size_mb,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "enable_gnn": settings.ENABLE_GNN_MODULE,
        "enable_composite": settings.ENABLE_COMPOSITE_MODULE,
        "enable_hardness": settings.ENABLE_HARDNESS_MODULE,
    }


# ── App Configuration ──────────────────────

def get_app_config() -> dict[str, str]:
    """Read safe .env values for display."""
    env = _read_env_file()
    safe_keys = [
        "APP_VERSION", "APP_NAME", "APP_LOGO_TEXT", "APP_LOGO_PATH",
        "APP_TAGLINE", "NEXT_PUBLIC_APP_VERSION", "NEXT_PUBLIC_APP_NAME",
        "NEXT_PUBLIC_APP_LOGO_TEXT", "NEXT_PUBLIC_APP_LOGO_PATH",
        "NEXT_PUBLIC_DEV_NAME", "NEXT_PUBLIC_DEV_GITHUB",
        "NEXT_PUBLIC_DEV_LINKEDIN", "DATABASE_URL", "CORS_ORIGINS",
        "MODEL_ARTIFACTS_PATH", "TRAINING_ARTIFACTS_PATH",
        "ENABLE_COMPOSITE_MODULE", "ENABLE_HARDNESS_MODULE",
        "ENABLE_GNN_MODULE",
    ]
    return {k: env.get(k, "") for k in safe_keys}


def update_app_config(updates: dict[str, str]) -> dict[str, str]:
    """Update .env file with new values. Returns updated config."""
    blocked = {"API_SECRET_KEY", "DATABASE_URL"}
    safe_updates = {k: v for k, v in updates.items() if k not in blocked}
    if safe_updates:
        _write_env_file(safe_updates)
    return get_app_config()


def import_env_file(content: str) -> dict[str, Any]:
    """Parse an uploaded .env file content and merge into the main .env."""
    blocked = {"API_SECRET_KEY"}
    env_vars: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip()

    if not env_vars:
        return {"success": False, "message": "No valid key-value pairs found in the file",
                "keys_updated": 0, "keys_added": 0, "keys_skipped": 0, "skipped_keys": []}

    current = _read_env_file()
    updates: dict[str, str] = {}
    skipped: list[str] = []
    added = 0
    updated = 0

    for key, value in env_vars.items():
        if key in blocked:
            skipped.append(key)
            continue
        if key in current:
            if current[key] != value:
                updates[key] = value
                updated += 1
        else:
            updates[key] = value
            added += 1

    if updates:
        _write_env_file(updates)

    return {
        "success": True,
        "message": f"Imported {updated + added} variables ({updated} updated, {added} added)",
        "keys_updated": updated,
        "keys_added": added,
        "keys_skipped": len(skipped),
        "skipped_keys": skipped,
    }


# ── LLM Configuration ──────────────────────

def get_llm_config() -> dict[str, Any]:
    """Get current LLM configuration."""
    env = _read_env_file()
    api_key = env.get("LLM_API_KEY", "") or env.get("GEMINI_API_KEY", "")
    provider = env.get("LLM_PROVIDER", "")
    model = env.get("LLM_MODEL", "")

    status = "not_configured"
    status_message = "No LLM provider configured"
    if provider and api_key:
        status = "ready"
        status_message = f"{provider}/{model}"
    elif provider and not api_key:
        status = "error"
        status_message = "API key missing"

    return {
        "provider": provider,
        "model": model,
        "has_api_key": bool(api_key),
        "base_url": env.get("LLM_BASE_URL", ""),
        "temperature": float(env.get("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(env.get("LLM_MAX_TOKENS", "4096")),
        "status": status,
        "status_message": status_message,
    }


def update_llm_config(data: dict[str, Any]) -> dict[str, Any]:
    """Update LLM settings in .env."""
    env_updates: dict[str, str] = {}
    mapping = {
        "provider": "LLM_PROVIDER",
        "model": "LLM_MODEL",
        "api_key": "LLM_API_KEY",
        "base_url": "LLM_BASE_URL",
        "temperature": "LLM_TEMPERATURE",
        "max_tokens": "LLM_MAX_TOKENS",
    }
    for field, env_key in mapping.items():
        if field in data and data[field] is not None:
            env_updates[env_key] = str(data[field])
            if field == "api_key" and data.get("provider") == "google":
                env_updates["GEMINI_API_KEY"] = str(data[field])

    if env_updates:
        _write_env_file(env_updates)
    return get_llm_config()


def get_llm_providers() -> list[dict[str, Any]]:
    """Return list of supported LLM providers."""
    return [
        {"id": "google", "name": "Google Gemini", "description": "Gemini Flash/Pro models",
         "requires_api_key": True, "requires_base_url": False,
         "default_models": ["gemini-3-flash-preview", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-04-17"],
         "icon": "sparkles"},
        {"id": "openai", "name": "OpenAI", "description": "GPT-4o, GPT-4.1, o3/o4-mini",
         "requires_api_key": True, "requires_base_url": False,
         "default_models": ["gpt-4o", "gpt-4.1", "o4-mini", "o3"],
         "icon": "bot"},
        {"id": "anthropic", "name": "Anthropic", "description": "Claude Sonnet/Opus models",
         "requires_api_key": True, "requires_base_url": False,
         "default_models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3.5-haiku-20241022"],
         "icon": "brain"},
        {"id": "deepseek", "name": "DeepSeek", "description": "DeepSeek R1/V3 models",
         "requires_api_key": True, "requires_base_url": True,
         "default_models": ["deepseek-chat", "deepseek-r1"],
         "icon": "search"},
        {"id": "ollama", "name": "Ollama (Local)", "description": "Local models via Ollama",
         "requires_api_key": False, "requires_base_url": True,
         "default_models": ["llama3.3", "qwen3", "mistral", "gemma3"],
         "icon": "server"},
        {"id": "custom", "name": "Custom Provider", "description": "Any OpenAI-compatible API",
         "requires_api_key": True, "requires_base_url": True,
         "default_models": [],
         "icon": "settings"},
    ]


# ── Element Registry ──────────────────────

# Default property keys (original, shipped with app)
DEFAULT_PROPERTY_KEYS = (
    "atomic_number", "symbol", "atomic_mass", "en_pauling",
    "atomic_radius_pm", "ionic_radius_pm", "covalent_radius_pm",
    "vdw_radius_pm", "melting_point_k", "boiling_point_k",
    "electron_affinity_ev", "ionization_energy_ev", "valence_electrons",
    "group", "period", "block", "density_g_cm3", "specific_heat_j_gk",
    "thermal_conductivity_w_mk", "bulk_modulus_gpa", "shear_modulus_gpa",
    "youngs_modulus_gpa", "poisson_ratio", "polarizability_a3",
    "oxidation_states", "coordination_number", "perovskite_site",
    "is_rare_earth",
)

# Default supported elements (original, shipped with app)
DEFAULT_SUPPORTED_ELEMENTS = frozenset({
    "K", "Na", "Li", "Ba", "Ca", "Sr", "Ag",
    "Bi", "Pb",
    "Nb", "Ta", "Ti", "Zr", "Hf", "Sb", "W", "Mo", "Sn", "Sc", "Fe",
    "Cu", "Mn", "Al", "Mg", "Zn",
    "La", "Nd", "Pr", "Sm", "Eu", "Gd", "Ho",
    "O",
})


def get_element_registry() -> dict[str, Any]:
    """Get full element registry state."""
    from piezo_ml.registry.element_registry import (
        ELEMENT_REGISTRY, PENDING_ELEMENTS, PROPERTY_KEYS,
        A_SITE, B_SITE, DOPANTS, RARE_EARTHS,
    )

    customs = _load_custom_additions()
    user_added_elems = set(customs.get("added_elements", []))
    user_added_props = customs.get("added_properties", [])
    elem_categories = customs.get("element_categories", {})  # user-stored categories

    elements = []
    for symbol, props in sorted(ELEMENT_REGISTRY.items()):
        # Check user-stored categories first for user-added elements
        stored_cats = elem_categories.get(symbol, [])
        if stored_cats:
            # Use the first user-selected category as primary
            cat = stored_cats[0].replace("_", "_")  # already in correct format
        elif symbol == "O":
            cat = "anion"
        elif symbol in A_SITE:
            cat = "A-site"
        elif symbol in B_SITE:
            cat = "B-site"
        elif symbol in RARE_EARTHS:
            cat = "rare_earth"
        elif symbol in DOPANTS:
            cat = "dopant"
        else:
            cat = "other"
        elements.append({
            "symbol": symbol,
            "atomic_number": props.get("atomic_number", 0),
            "category": cat,
            "perovskite_site": props.get("perovskite_site", ""),
            "is_rare_earth": props.get("is_rare_earth", False),
            "is_pending": False,
            "is_user_added": symbol in user_added_elems,
            "property_count": sum(1 for v in props.values() if v is not None),
        })

    # Combine default + user-added property keys
    all_props = list(PROPERTY_KEYS) + [p for p in user_added_props if p not in PROPERTY_KEYS]

    return {
        "supported_elements": elements,
        "pending_elements": list(PENDING_ELEMENTS),
        "user_added_elements": sorted(user_added_elems),
        "total_properties": len(all_props),
        "property_keys": all_props,
        "default_property_keys": list(DEFAULT_PROPERTY_KEYS),
        "user_added_properties": user_added_props,
    }


def add_pending_element(symbol: str, categories: list[str]) -> dict[str, Any]:
    """Add element to PENDING_ELEMENTS list and store categories."""
    from piezo_ml.registry.element_registry import (
        PENDING_ELEMENTS, SUPPORTED_ELEMENTS, ELEMENT_REGISTRY,
    )
    symbol = symbol.strip()
    # Validate symbol format: first char uppercase, rest lowercase, 1-3 chars
    if not symbol or len(symbol) > 3:
        return {"error": f"Invalid symbol: '{symbol}'. Must be 1-3 characters."}
    if not symbol[0].isupper():
        return {"error": f"Symbol must start with an uppercase letter (e.g., Na, not na)."}
    if len(symbol) > 1 and not symbol[1:].islower():
        return {"error": f"Only the first character should be uppercase (e.g., Na, not NA)."}
    if symbol in SUPPORTED_ELEMENTS or symbol in ELEMENT_REGISTRY:
        return {"error": f"{symbol} is already supported"}
    if symbol in PENDING_ELEMENTS:
        return {"error": f"{symbol} is already pending"}
    PENDING_ELEMENTS.append(symbol)

    # Store categories for this element
    customs = _load_custom_additions()
    elem_cats = customs.get("element_categories", {})
    elem_cats[symbol] = categories
    customs["element_categories"] = elem_cats
    _save_custom_additions(customs)

    return {"message": f"{symbol} added to pending elements", "symbol": symbol}


def remove_pending_element(symbol: str) -> dict[str, Any]:
    """Remove element from PENDING_ELEMENTS list."""
    from piezo_ml.registry.element_registry import PENDING_ELEMENTS
    if symbol in PENDING_ELEMENTS:
        PENDING_ELEMENTS.remove(symbol)
        return {"message": f"{symbol} removed from pending"}
    return {"error": f"{symbol} not in pending list"}


def remove_supported_element(symbol: str) -> dict[str, Any]:
    """Remove a user-added element from the supported registry."""
    customs = _load_custom_additions()
    user_elems = customs.get("added_elements", [])
    if symbol not in user_elems:
        return {"error": f"{symbol} is a default element and cannot be removed. Only user-added elements can be removed."}

    from piezo_ml.registry.element_registry import ELEMENT_REGISTRY, REGISTRY_DATA_PATH, _load_registry_file

    # Remove from in-memory registry
    if symbol in ELEMENT_REGISTRY:
        del ELEMENT_REGISTRY[symbol]

    # Remove from persisted registry
    registry = _load_registry_file()
    if symbol in registry:
        del registry[symbol]
        with open(REGISTRY_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)

    # Remove from tracking
    user_elems.remove(symbol)
    customs["added_elements"] = user_elems
    _save_custom_additions(customs)

    return {"message": f"{symbol} removed from registry"}


def bootstrap_elements() -> dict[str, Any]:
    """Run bootstrap for pending elements."""
    from piezo_ml.registry.element_registry import (
        PENDING_ELEMENTS,
        ELEMENT_REGISTRY, REGISTRY_DATA_PATH,
        _load_registry_file, _build_from_sources,
    )

    if not PENDING_ELEMENTS:
        return {"bootstrapped": [], "failed": [], "message": "No pending elements to bootstrap"}

    registry = _load_registry_file()
    bootstrapped = []
    failed = []

    for symbol in list(PENDING_ELEMENTS):
        if symbol in registry:
            # Already in registry file — just add to in-memory
            ELEMENT_REGISTRY[symbol] = registry[symbol]
            PENDING_ELEMENTS.remove(symbol)
            bootstrapped.append(symbol)
            continue
        try:
            data = _build_from_sources(symbol)
            registry[symbol] = data
            ELEMENT_REGISTRY[symbol] = data
            PENDING_ELEMENTS.remove(symbol)
            bootstrapped.append(symbol)
        except Exception as e:
            logger.warning(f"Failed to bootstrap {symbol}: {e}")
            failed.append(symbol)

    # Persist to registry data file
    if bootstrapped:
        with open(REGISTRY_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)

    # Track user-added elements
    if bootstrapped:
        customs = _load_custom_additions()
        existing_added = set(customs.get("added_elements", []))
        for sym in bootstrapped:
            existing_added.add(sym)
        customs["added_elements"] = sorted(existing_added)
        _save_custom_additions(customs)

    parts = []
    if bootstrapped:
        parts.append(f"Bootstrapped {len(bootstrapped)}: {', '.join(bootstrapped)}")
    if failed:
        parts.append(f"Failed {len(failed)}: {', '.join(failed)} (invalid symbol?)")
    message = ". ".join(parts) if parts else "No pending elements to bootstrap"

    return {
        "bootstrapped": bootstrapped,
        "failed": failed,
        "message": message,
    }


# ── Custom Properties ──────────────────────

def add_custom_property(property_key: str) -> dict[str, Any]:
    """Add a custom property key. NOTE: This is metadata-only; the property will need
    to be added to the registry data file per-element. The frontend should inform the
    user that a server restart or re-bootstrap may be needed for full effect."""
    import re
    # Validate snake_case format
    if not re.match(r'^[a-z][a-z0-9_]*$', property_key):
        return {"error": "Property key must be snake_case (e.g., dielectric_constant)"}

    customs = _load_custom_additions()
    user_props = customs.get("added_properties", [])

    from piezo_ml.registry.element_registry import PROPERTY_KEYS
    if property_key in PROPERTY_KEYS:
        return {"error": f"'{property_key}' is already a default property"}
    if property_key in user_props:
        return {"error": f"'{property_key}' is already added"}

    user_props.append(property_key)
    customs["added_properties"] = user_props
    _save_custom_additions(customs)

    return {"message": f"Property '{property_key}' added. Values will default to null for all elements until populated.",
            "property_key": property_key}


def remove_custom_property(property_key: str) -> dict[str, Any]:
    """Remove a user-added custom property."""
    customs = _load_custom_additions()
    user_props = customs.get("added_properties", [])

    if property_key not in user_props:
        return {"error": f"'{property_key}' is a default property and cannot be removed"}

    user_props.remove(property_key)
    customs["added_properties"] = user_props
    _save_custom_additions(customs)

    return {"message": f"Property '{property_key}' removed"}


# ── Reset ──────────────────────

def reset_elements_and_properties() -> dict[str, Any]:
    """Reset elements and properties to defaults. Removes all user-added items."""
    from piezo_ml.registry.element_registry import (
        ELEMENT_REGISTRY, REGISTRY_DATA_PATH, _load_registry_file,
        PENDING_ELEMENTS, SUPPORTED_ELEMENTS,
    )

    customs = _load_custom_additions()
    removed_elems = customs.get("added_elements", [])
    removed_props = customs.get("added_properties", [])
    actions = []

    # Remove user-added elements from in-memory and persisted registry
    for sym in removed_elems:
        if sym in ELEMENT_REGISTRY:
            del ELEMENT_REGISTRY[sym]
    if removed_elems:
        registry = _load_registry_file()
        for sym in removed_elems:
            registry.pop(sym, None)
        with open(REGISTRY_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)
        actions.append(f"Removed {len(removed_elems)} user-added elements: {', '.join(removed_elems)}")

    # Clear pending
    PENDING_ELEMENTS.clear()
    actions.append("Cleared pending elements list")

    if removed_props:
        actions.append(f"Removed {len(removed_props)} user-added properties: {', '.join(removed_props)}")

    # Reset customizations file
    _save_custom_additions({"added_elements": [], "added_properties": []})

    return {"success": True, "message": "Elements and properties reset to defaults",
            "actions_taken": actions}


def reset_all_settings() -> dict[str, Any]:
    """Factory reset: restore .env to defaults, clear customizations, clear localStorage hint."""
    actions = []

    # 1. Reset elements and properties
    elem_result = reset_elements_and_properties()
    actions.extend(elem_result.get("actions_taken", []))

    # 2. Restore .env to defaults
    if DEFAULT_ENV_BACKUP_PATH.exists():
        shutil.copy2(DEFAULT_ENV_BACKUP_PATH, ENV_PATH)
        actions.append("Restored .env to original defaults")
    elif ENV_EXAMPLE_PATH.exists():
        shutil.copy2(ENV_EXAMPLE_PATH, ENV_PATH)
        actions.append("Restored .env from .env.example")
    else:
        actions.append("Warning: No default .env backup found — .env unchanged")

    # 3. Clear customizations
    if CUSTOM_ADDITIONS_PATH.exists():
        os.remove(CUSTOM_ADDITIONS_PATH)
        actions.append("Cleared settings customizations file")

    return {
        "success": True,
        "message": "All settings reset to factory defaults. Restart the server for full effect.",
        "actions_taken": actions,
    }


# ── Danger Zone ──────────────────────

async def purge_all_models(db: AsyncSession) -> dict[str, Any]:
    """Delete ALL trained models from DB and filesystem."""
    from piezo_db.models import TrainedModel, TrainingJob

    count = (await db.execute(select(func.count(TrainedModel.id)))).scalar() or 0

    await db.execute(text("DELETE FROM predictions"))
    await db.execute(text("DELETE FROM prediction_batches"))
    await db.execute(text("DELETE FROM trained_models"))
    await db.execute(text("DELETE FROM training_jobs"))
    await db.commit()

    models_dir = PROJECT_ROOT / settings.MODEL_ARTIFACTS_PATH.lstrip("./")
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_file():
                item.unlink()

    artifacts_dir = PROJECT_ROOT / settings.TRAINING_ARTIFACTS_PATH.lstrip("./")
    if artifacts_dir.exists():
        for item in artifacts_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

    return {"success": True, "message": f"Purged {count} models", "items_affected": count}


async def clear_prediction_cache(db: AsyncSession) -> dict[str, Any]:
    """Clear all predictions and batch data."""
    from piezo_db.models import Prediction, PredictionBatch

    pred_count = (await db.execute(select(func.count(Prediction.id)))).scalar() or 0
    batch_count = (await db.execute(select(func.count(PredictionBatch.id)))).scalar() or 0

    await db.execute(text("DELETE FROM predictions"))
    await db.execute(text("DELETE FROM prediction_batches"))
    await db.commit()

    results_dir = PROJECT_ROOT / "resources" / "prediction-results"
    if results_dir.exists():
        shutil.rmtree(results_dir, ignore_errors=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    total = pred_count + batch_count
    return {
        "success": True,
        "message": f"Cleared {pred_count} predictions and {batch_count} batches",
        "items_affected": total,
    }


async def batch_delete_models(db: AsyncSession, model_ids: list[str]) -> dict[str, Any]:
    """Delete multiple models by their IDs."""
    from piezo_db.models import TrainedModel

    deleted = 0
    for mid in model_ids:
        try:
            uid = UUID(mid)
        except ValueError:
            continue
        model = await db.get(TrainedModel, uid)
        if model:
            model_path = PROJECT_ROOT / model.model_file_path.lstrip("./")
            if model_path.exists():
                model_path.unlink()
            await db.delete(model)
            deleted += 1

    await db.commit()
    return {"success": True, "message": f"Deleted {deleted} models", "items_affected": deleted}


# ── GNN Status ──────────────────────

def get_gnn_status() -> dict[str, Any]:
    """Check GNN/CHGNet dependency status."""
    enabled = settings.ENABLE_GNN_MODULE
    installed = False
    pytorch_version = ""
    chgnet_version = ""

    try:
        import torch
        pytorch_version = torch.__version__
        installed = True
    except ImportError:
        pass

    try:
        import chgnet
        chgnet_version = getattr(chgnet, "__version__", "installed")
    except ImportError:
        if installed:
            chgnet_version = "not installed"

    if not installed:
        msg = "PyTorch and CHGNet are not installed."
        instructions = (
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            "pip install chgnet matgl"
        )
    elif not chgnet_version or chgnet_version == "not installed":
        msg = f"PyTorch {pytorch_version} installed but CHGNet is missing."
        instructions = "pip install chgnet matgl"
    else:
        msg = f"GNN ready — PyTorch {pytorch_version}, CHGNet {chgnet_version}"
        instructions = ""

    return {
        "enabled": enabled,
        "installed": installed and bool(chgnet_version) and chgnet_version != "not installed",
        "pytorch_version": pytorch_version,
        "chgnet_version": chgnet_version,
        "install_instructions": instructions,
        "message": msg,
    }
