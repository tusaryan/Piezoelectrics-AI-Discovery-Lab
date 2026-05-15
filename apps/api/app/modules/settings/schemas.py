"""
Piezo.AI — Settings Module Schemas
===================================
Pydantic models for settings endpoints.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── System Environment ──────────────────────
class SystemEnvironment(BaseModel):
    """System overview stats."""
    app_name: str
    app_version: str
    dataset_count: int = 0
    total_rows: int = 0
    trained_model_count: int = 0
    prediction_count: int = 0
    db_size_mb: float = 0.0
    python_version: str = ""
    enable_gnn: bool = False
    enable_composite: bool = True
    enable_hardness: bool = True


# ── App Configuration ──────────────────────
class AppConfigResponse(BaseModel):
    """Current .env configuration values (safe subset)."""
    APP_VERSION: str = ""
    APP_NAME: str = ""
    APP_LOGO_TEXT: str = ""
    APP_LOGO_PATH: str = ""
    APP_TAGLINE: str = ""
    NEXT_PUBLIC_APP_VERSION: str = ""
    NEXT_PUBLIC_APP_NAME: str = ""
    NEXT_PUBLIC_APP_LOGO_TEXT: str = ""
    NEXT_PUBLIC_APP_LOGO_PATH: str = ""
    NEXT_PUBLIC_DEV_NAME: str = ""
    NEXT_PUBLIC_DEV_GITHUB: str = ""
    NEXT_PUBLIC_DEV_LINKEDIN: str = ""
    DATABASE_URL: str = ""
    CORS_ORIGINS: str = ""
    MODEL_ARTIFACTS_PATH: str = ""
    TRAINING_ARTIFACTS_PATH: str = ""
    ENABLE_COMPOSITE_MODULE: str = ""
    ENABLE_HARDNESS_MODULE: str = ""
    ENABLE_GNN_MODULE: str = ""


class AppConfigUpdate(BaseModel):
    """Update .env variables — partial update."""
    updates: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs to write to .env",
    )


# ── LLM Configuration ──────────────────────
class LlmConfigResponse(BaseModel):
    """Current LLM configuration."""
    provider: str = ""
    model: str = ""
    has_api_key: bool = False
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    status: str = "not_configured"  # not_configured | ready | error
    status_message: str = ""


class LlmConfigUpdate(BaseModel):
    """Update LLM settings."""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class LlmProviderInfo(BaseModel):
    """Information about a supported LLM provider."""
    id: str
    name: str
    description: str
    requires_api_key: bool = True
    requires_base_url: bool = False
    default_models: list[str] = Field(default_factory=list)
    icon: str = ""


# ── Element Registry ──────────────────────
class ElementInfo(BaseModel):
    """Element in the registry."""
    symbol: str
    atomic_number: int = 0
    category: str = ""  # A-site, B-site, dopant, rare_earth, anion
    perovskite_site: str = ""
    is_rare_earth: bool = False
    is_pending: bool = False
    is_user_added: bool = False
    property_count: int = 0


class PendingElementRequest(BaseModel):
    """Add a new pending element."""
    symbol: str = Field(..., min_length=1, max_length=3)
    categories: list[str] = Field(
        default_factory=list,
        description="Site assignments: A-site, B-site, dopant",
    )


class ElementRegistryResponse(BaseModel):
    """Full element registry state."""
    supported_elements: list[ElementInfo] = Field(default_factory=list)
    pending_elements: list[str] = Field(default_factory=list)
    user_added_elements: list[str] = Field(default_factory=list)
    total_properties: int = 0
    property_keys: list[str] = Field(default_factory=list)
    default_property_keys: list[str] = Field(default_factory=list)
    user_added_properties: list[str] = Field(default_factory=list)


class BootstrapResult(BaseModel):
    """Result of bootstrapping pending elements."""
    bootstrapped: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    message: str = ""


# ── Custom Property ──────────────────────
class AddPropertyRequest(BaseModel):
    """Add a new custom property key to the registry."""
    property_key: str = Field(..., min_length=1, max_length=100,
        description="Snake_case property name, e.g., 'dielectric_constant'")


class RemovePropertyRequest(BaseModel):
    """Remove a custom property key."""
    property_key: str = Field(..., min_length=1, max_length=100)


class RemoveElementRequest(BaseModel):
    """Remove a user-added element from the registry."""
    symbol: str = Field(..., min_length=1, max_length=3)


# ── Field Schema Manager ──────────────────────
class FieldDefinitionResponse(BaseModel):
    """A single field definition from the schema."""
    name: str
    data_type: str  # float | int | string | category
    description: str = ""
    is_target: bool = False
    is_input: bool = True
    is_required: bool = False
    is_composite_field: bool = False
    category_values: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    default_value: Optional[str] = None
    is_user_added: bool = False
    added_at: Optional[str] = None


class AddFieldRequest(BaseModel):
    """Add a new user-defined field."""
    name: str = Field(..., min_length=1, max_length=100,
        description="Snake_case field name, e.g., 'dielectric_constant'")
    data_type: str = Field(..., description="float | int | string | category")
    description: str = ""
    is_target: bool = False
    is_input: bool = True
    is_required: bool = False
    is_composite_field: bool = False
    category_values: list[str] = Field(default_factory=list)
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    default_value: Optional[str] = None


class AddCategoryRequest(BaseModel):
    """Add a category value to an existing field."""
    value: str = Field(..., min_length=1, max_length=100)


class AddAliasRequest(BaseModel):
    """Add an alias mapping for a field."""
    alias: str = Field(..., min_length=1, max_length=100)
    canonical: str = Field(..., min_length=1, max_length=100)


class FieldSchemaImportRequest(BaseModel):
    """Import field schema customizations."""
    schema_data: dict = Field(..., description="Exported schema JSON data")


# ── Model Library ──────────────────────
class SettingsModelInfo(BaseModel):
    """Model info for settings library."""
    id: str
    display_name: str
    target: str
    algorithm: str
    r2_score: float
    rmse: float
    feature_dim: int = 0
    n_train_samples: int = 0
    n_test_samples: int = 0
    is_default: bool = False
    model_file_path: str = ""
    created_at: str = ""


class ModelRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1, max_length=255)


class BatchDeleteRequest(BaseModel):
    """Batch delete models by IDs."""
    model_ids: list[str] = Field(..., min_length=1)


# ── Danger Zone ──────────────────────
class DangerActionRequest(BaseModel):
    """Confirmation for danger zone actions."""
    confirm: bool = False


class DangerActionResponse(BaseModel):
    """Result of a danger zone action."""
    success: bool
    message: str
    items_affected: int = 0


# ── GNN Status ──────────────────────
class GnnStatusResponse(BaseModel):
    """GNN/CHGNet dependency status."""
    enabled: bool = False
    installed: bool = False
    pytorch_version: str = ""
    chgnet_version: str = ""
    install_instructions: str = ""
    message: str = ""


# ── Import Env ──────────────────────
class EnvImportResponse(BaseModel):
    """Result of importing .env file."""
    success: bool
    message: str
    keys_updated: int = 0
    keys_added: int = 0
    keys_skipped: int = 0
    skipped_keys: list[str] = Field(default_factory=list)


# ── Reset ──────────────────────
class ResetResponse(BaseModel):
    """Result of reset action."""
    success: bool
    message: str
    actions_taken: list[str] = Field(default_factory=list)
