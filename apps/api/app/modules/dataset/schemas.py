"""
Piezo.AI — Dataset Module Pydantic Schemas
============================================
Request/response models for dataset upload, column mapping,
material CRUD, and data quality reporting.

See s2-plan-backend.md for full specification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Backend field definitions (single source of truth for valid field names)
# ---------------------------------------------------------------------------

# All valid backend field names that CSV columns can be mapped to
BACKEND_FIELDS: dict[str, list[str]] = {
    # Required
    "formula": ["formula", "composition", "chemical_formula", "chem_formula", "chemical"],
    # Core numeric targets
    "d33": ["d33", "d_33", "piezo_coeff", "piezoelectric_coefficient", "piezoelectric"],
    "tc": ["tc", "t_c", "curie_temp", "curie_temperature", "curie"],
    "vickers_hardness": ["vickers_hardness", "hardness", "hv", "vickers", "vickers_hv"],
    # Additional numeric properties
    "qm": ["qm", "q_m", "quality_factor", "mechanical_quality"],
    "kp": ["kp", "k_p", "coupling", "coupling_coefficient"],
    "relative_density_pct": ["relative_density", "density_pct", "density", "rel_density", "relative_density_pct"],
    "sintering_temp_c": ["sintering_temp", "sinter_temp", "sintering_temperature", "sintering_temp_c"],
    # Categorical
    "sintering_method": ["sintering_method", "sinter_method"],
    "ceramic_type": ["ceramic_type", "type", "material_type"],
    "fabrication_method": ["fabrication_method", "fab_method", "fabrication"],
    # Composite fields
    "matrix_type": ["matrix_type", "matrix", "polymer_matrix"],
    "filler_wt_pct": ["filler_wt_pct", "filler_pct", "filler_weight", "wt_pct", "filler"],
    "particle_morphology": ["particle_morphology", "morphology", "particle_shape"],
    "particle_size_nm": ["particle_size_nm", "particle_size", "size_nm"],
    "surface_treatment": ["surface_treatment", "treatment", "surface_modification"],
    # Traceability
    "source_doi": ["source_doi", "doi", "reference", "source"],
    "source_notes": ["source_notes", "notes", "comments"],
}

# Fields that have constrained valid values (for dropdown rendering)
# NOTE: These are SUGGESTIONS, not strict constraints. During quality report,
# values outside these lists are flagged as warnings, not errors.
CATEGORICAL_FIELD_OPTIONS: dict[str, list[str]] = {
    "sintering_method": [
        "conventional", "hot_press", "sps", "rtgg", "tgg", "two_step",
        "cold_sinter", "microwave", "flash",
    ],
    "ceramic_type": ["soft", "hard", "composite"],
    "fabrication_method": [
        "conventional", "hot_press", "sps", "rtgg", "tgg", "two_step",
        "electrospinning", "solvent_cast", "cold_sinter", "hot_compression",
        "3d_print", "tape_casting", "screen_printing", "injection_molding",
    ],
    "matrix_type": [
        "none", "pvdf", "p_vdf_trfe", "pvdf_trfe", "pvdf_hfp",
        "pvdf_hfp_ctrfe", "epoxy", "silicone", "polyimide", "pdms",
    ],
    "particle_morphology": [
        "none", "spherical", "rod", "cube", "nanoblock", "fiber",
        "platelet", "whisker", "irregular",
    ],
    "surface_treatment": [
        "none", "untreated", "silane", "plasma", "acid", "peg",
        "dopamine", "fluorinated", "kh550", "kh560",
    ],
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ColumnMappingRequest(BaseModel):
    """Apply column mapping from CSV column names to backend field names."""
    mapping: dict[str, str] = Field(
        ...,
        description="Mapping from CSV column names to backend field names. "
                    "e.g. {'Chemical Formula': 'formula', 'd33 (pC/N)': 'd33'}",
    )

    @field_validator("mapping")
    @classmethod
    def formula_must_be_mapped(cls, v: dict[str, str]) -> dict[str, str]:
        """formula is mandatory — must appear in the mapping values."""
        mapped_fields = set(v.values())
        if "formula" not in mapped_fields:
            raise ValueError(
                "Column mapping must include 'formula' — "
                "it is required for all datasets."
            )
        # Validate all target fields are valid backend field names
        valid_fields = set(BACKEND_FIELDS.keys())
        invalid = mapped_fields - valid_fields
        if invalid:
            raise ValueError(
                f"Invalid backend field names: {', '.join(sorted(invalid))}. "
                f"Valid fields: {', '.join(sorted(valid_fields))}"
            )
        return v


class MaterialUpdateRequest(BaseModel):
    """Update fields on a single material row."""
    formula: str | None = None
    d33: float | None = None
    tc: float | None = None
    vickers_hardness: float | None = None
    qm: float | None = None
    kp: float | None = None
    relative_density_pct: float | None = None
    sintering_temp_c: float | None = None
    sintering_method: str | None = None
    ceramic_type: str | None = None
    fabrication_method: str | None = None
    matrix_type: str | None = None
    filler_wt_pct: float | None = None
    particle_morphology: str | None = None
    particle_size_nm: float | None = None
    surface_treatment: str | None = None
    source_doi: str | None = None
    source_notes: str | None = None


class MaterialCreateRequest(BaseModel):
    """Add a new row to a dataset."""
    formula: str
    d33: float | None = None
    tc: float | None = None
    vickers_hardness: float | None = None
    qm: float | None = None
    kp: float | None = None
    relative_density_pct: float | None = None
    sintering_temp_c: float | None = None
    sintering_method: str | None = None
    ceramic_type: str | None = None
    fabrication_method: str | None = None
    matrix_type: str = "none"
    filler_wt_pct: float = 0
    particle_morphology: str = "none"
    particle_size_nm: float | None = None
    surface_treatment: str = "none"
    source_doi: str | None = None
    source_notes: str | None = None


class BulkUpdateItem(BaseModel):
    """Single item in a bulk update request."""
    id: str
    updates: dict[str, Any]


class BulkMaterialRequest(BaseModel):
    """Bulk update and/or delete materials."""
    updates: list[BulkUpdateItem] = Field(default_factory=list)
    deletes: list[str] = Field(
        default_factory=list,
        description="List of material UUIDs to delete",
    )


class DatasetRenameRequest(BaseModel):
    """Rename a dataset (display_name only — UUID never changes)."""
    display_name: str = Field(..., min_length=1, max_length=255)


class DatasetCopyRequest(BaseModel):
    """Copy a dataset with all materials."""
    new_name: str | None = Field(None, max_length=255, description="Name for the copy (optional)")


class BulkDeleteDatasetsRequest(BaseModel):
    """Delete multiple datasets at once."""
    dataset_ids: list[str] = Field(..., min_length=1, description="Dataset UUIDs to delete")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class DatasetSummaryResponse(BaseModel):
    """Lightweight dataset summary for listing."""
    id: str
    display_name: str
    original_filename: str
    status: str
    total_rows: int
    total_columns: int
    has_composite_fields: bool
    uploaded_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class DatasetDetailResponse(DatasetSummaryResponse):
    """Full dataset detail including column mapping."""
    column_mapping: dict[str, str]


class MaterialResponse(BaseModel):
    """Single material row response."""
    id: str
    uid: int
    formula: str
    d33: float | None = None
    tc: float | None = None
    vickers_hardness: float | None = None
    qm: float | None = None
    kp: float | None = None
    relative_density_pct: float | None = None
    sintering_temp_c: float | None = None
    sintering_method: str | None = None
    ceramic_type: str | None = None
    fabrication_method: str | None = None
    matrix_type: str = "none"
    filler_wt_pct: float = 0
    particle_morphology: str = "none"
    particle_size_nm: float | None = None
    surface_treatment: str = "none"
    source_doi: str | None = None
    source_notes: str | None = None
    parse_status: str = "pending"
    parse_warnings: str | None = None
    source_row: dict[str, Any] | None = None
    parsed_row: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class UploadPreviewResponse(BaseModel):
    """Returned after CSV upload, before column mapping."""
    dataset_id: str
    filename: str
    columns: list[str]
    row_count: int
    preview_rows: list[dict[str, Any]]
    suggested_mapping: dict[str, str]


class DataIssue(BaseModel):
    """Single data quality issue."""
    row_uid: int
    material_id: str
    column: str
    issue_type: str  # missing_value | type_mismatch | unsupported_element | invalid_character | parse_error
    current_value: str | None = None
    message: str


class ColumnStats(BaseModel):
    """Per-column quality statistics."""
    column_name: str
    total_values: int
    missing_count: int
    invalid_count: int
    dtype: str


class DataQualityReport(BaseModel):
    """Data quality report after column mapping + validation."""
    total_rows: int
    valid_rows: int
    issue_count: int
    issues: list[DataIssue]
    column_stats: list[ColumnStats]


class PaginatedMaterialsResponse(BaseModel):
    """Paginated materials list."""
    items: list[MaterialResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkUpdateResponse(BaseModel):
    """Response from bulk update/delete."""
    updated_count: int
    deleted_count: int
    errors: list[str] = Field(default_factory=list)


class BackendFieldInfo(BaseModel):
    """Information about a backend field for the mapping UI."""
    name: str
    category: str  # required | core | processing | composite | traceability
    label: str
    description: str
    type: str  # string | float | int
    required: bool = False
    options: list[str] | None = None  # For categorical fields


# ---------------------------------------------------------------------------
# Field metadata for frontend rendering
# ---------------------------------------------------------------------------

BACKEND_FIELD_META: list[BackendFieldInfo] = [
    # Required
    BackendFieldInfo(name="formula", category="required", label="Formula",
                     description="Chemical composition formula", type="string", required=True),
    # Core properties
    BackendFieldInfo(name="d33", category="core", label="d₃₃ (pC/N)",
                     description="Piezoelectric coefficient", type="float"),
    BackendFieldInfo(name="tc", category="core", label="Tc (°C)",
                     description="Curie temperature", type="float"),
    BackendFieldInfo(name="vickers_hardness", category="core", label="Vickers Hardness (HV)",
                     description="Vickers hardness in kgf/mm²", type="float"),
    BackendFieldInfo(name="qm", category="core", label="Qm",
                     description="Mechanical quality factor", type="float"),
    BackendFieldInfo(name="kp", category="core", label="kp",
                     description="Planar coupling coefficient", type="float"),
    BackendFieldInfo(name="relative_density_pct", category="core", label="Relative Density (%)",
                     description="Percentage of theoretical density", type="float"),
    # Processing
    BackendFieldInfo(name="sintering_temp_c", category="processing", label="Sintering Temp (°C)",
                     description="Peak sintering temperature", type="float"),
    BackendFieldInfo(name="sintering_method", category="processing", label="Sintering Method",
                     description="Method used for sintering", type="string",
                     options=CATEGORICAL_FIELD_OPTIONS["sintering_method"]),
    BackendFieldInfo(name="ceramic_type", category="processing", label="Ceramic Type",
                     description="Soft, hard, or composite", type="string",
                     options=CATEGORICAL_FIELD_OPTIONS["ceramic_type"]),
    BackendFieldInfo(name="fabrication_method", category="processing", label="Fabrication Method",
                     description="Fabrication process used", type="string",
                     options=CATEGORICAL_FIELD_OPTIONS["fabrication_method"]),
    # Composite
    BackendFieldInfo(name="matrix_type", category="composite", label="Matrix Type",
                     description="Polymer matrix type", type="string",
                     options=CATEGORICAL_FIELD_OPTIONS["matrix_type"]),
    BackendFieldInfo(name="filler_wt_pct", category="composite", label="Filler Wt%",
                     description="Weight fraction of ceramic filler", type="float"),
    BackendFieldInfo(name="particle_morphology", category="composite", label="Particle Morphology",
                     description="Shape of filler particles", type="string",
                     options=CATEGORICAL_FIELD_OPTIONS["particle_morphology"]),
    BackendFieldInfo(name="particle_size_nm", category="composite", label="Particle Size (nm)",
                     description="Filler particle size in nanometers", type="float"),
    BackendFieldInfo(name="surface_treatment", category="composite", label="Surface Treatment",
                     description="Filler surface modification", type="string",
                     options=CATEGORICAL_FIELD_OPTIONS["surface_treatment"]),
    # Traceability
    BackendFieldInfo(name="source_doi", category="traceability", label="Source DOI",
                     description="DOI or reference URL", type="string"),
    BackendFieldInfo(name="source_notes", category="traceability", label="Source Notes",
                     description="Free text notes about data source", type="string"),
]
