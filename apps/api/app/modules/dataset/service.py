"""
Piezo.AI — Dataset Service Layer
===================================
Business logic for dataset upload, column mapping, material CRUD,
and data quality reporting.

ARCHITECTURAL RULE: This is part of the DUMB PIPE. It handles CSV I/O
and DB operations. Formula validation is delegated to ml-core.
"""

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from piezo_db.models import Dataset, Material
from piezo_ml.validators.formula_validator import validate_formulas_batch

from app.modules.dataset.schemas import (
    BACKEND_FIELDS,
    CATEGORICAL_FIELD_OPTIONS,
    BulkUpdateResponse,
    ColumnStats,
    DataIssue,
    DataQualityReport,
    DatasetDetailResponse,
    DatasetSummaryResponse,
    MaterialResponse,
    PaginatedMaterialsResponse,
    UploadPreviewResponse,
)

# ---------------------------------------------------------------------------
# Temp file storage for uploaded CSVs (before column mapping)
# ---------------------------------------------------------------------------

# Project root — used for temp file storage
_PROJECT_ROOT = Path(__file__).resolve().parents[5]  # apps/api/app/modules/dataset/ → root
_TEMP_DIR = _PROJECT_ROOT / "resources" / "training-artifacts" / ".tmp"


def _ensure_temp_dir() -> Path:
    """Create temp directory if it doesn't exist."""
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return _TEMP_DIR


def _get_temp_csv_path(dataset_id: str) -> Path:
    """Get the temp file path for a dataset's raw CSV."""
    return _ensure_temp_dir() / f"{dataset_id}.csv"


def _cleanup_temp_csv(dataset_id: str) -> None:
    """Remove temp CSV file after column mapping is applied."""
    path = _get_temp_csv_path(dataset_id)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Column mapping auto-suggestion
# ---------------------------------------------------------------------------

def suggest_column_mapping(csv_columns: list[str]) -> dict[str, str]:
    """
    Auto-suggest mapping from CSV column names to backend field names.

    Uses case-insensitive matching against known aliases.
    Returns {csv_column_name: backend_field_name} for matches found.
    """
    mapping: dict[str, str] = {}
    used_backend_fields: set[str] = set()

    for csv_col in csv_columns:
        csv_lower = csv_col.strip().lower().replace(" ", "_").replace("-", "_")

        for backend_field, aliases in BACKEND_FIELDS.items():
            if backend_field in used_backend_fields:
                continue
            # Check exact match with aliases
            if csv_lower in [a.lower() for a in aliases]:
                mapping[csv_col] = backend_field
                used_backend_fields.add(backend_field)
                break
            # Check if csv column name contains the backend field name
            if backend_field in csv_lower or csv_lower in backend_field:
                mapping[csv_col] = backend_field
                used_backend_fields.add(backend_field)
                break

    return mapping


# ---------------------------------------------------------------------------
# Bulk ceramic defaults
# ---------------------------------------------------------------------------

def apply_bulk_ceramic_defaults(row: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce sentinel values for bulk ceramic rows.

    Rule: If filler_wt_pct is 0 or missing, the row is a bulk ceramic
    and composite-specific fields must use standard defaults.
    """
    filler = row.get("filler_wt_pct")
    try:
        filler_val = float(filler) if filler is not None else 0
    except (ValueError, TypeError):
        filler_val = 0

    if filler_val == 0:
        row["matrix_type"] = "none"
        row["particle_morphology"] = "none"
        row["surface_treatment"] = "none"
        row["particle_size_nm"] = None
        row["filler_wt_pct"] = 0

    return row


def validate_material_consistency(row: dict[str, Any]) -> list[str]:
    """
    Validate bulk-vs-composite consistency for user edits.
    Domain rule:
    - Bulk: filler_wt_pct == 0 and matrix_type == 'none'
    - Composite: filler_wt_pct > 0 and matrix_type != 'none'
      and composite descriptor fields must be provided.
    """
    errors: list[str] = []

    filler_raw = row.get("filler_wt_pct")
    try:
        filler = float(filler_raw) if filler_raw is not None else 0.0
    except (TypeError, ValueError):
        errors.append("filler_wt_pct must be numeric.")
        return errors

    matrix = str(row.get("matrix_type") or "none").strip().lower()
    morphology = str(row.get("particle_morphology") or "none").strip().lower()
    treatment = str(row.get("surface_treatment") or "none").strip().lower()
    size_nm = row.get("particle_size_nm")

    if filler < 0:
        errors.append("filler_wt_pct cannot be negative.")

    is_bulk = filler == 0 and matrix == "none"
    is_composite = filler > 0 and matrix != "none"

    if not is_bulk and not is_composite:
        errors.append(
            f"Bulk/composite mismatch: matrix_type='{matrix}', filler_wt_pct={filler}. "
            "Use (matrix='none', filler_wt_pct=0) for bulk or (matrix!='none', filler_wt_pct>0) for composites."
        )
        return errors

    if is_composite:
        if morphology in ("", "none"):
            errors.append("Composite rows require particle_morphology (not 'none').")
        if treatment in ("", "none"):
            errors.append("Composite rows require surface_treatment (use 'untreated' if no treatment was applied).")
        if size_nm is None:
            errors.append("Composite rows require particle_size_nm.")
        else:
            try:
                if float(size_nm) <= 0:
                    errors.append("particle_size_nm must be > 0 for composite rows.")
            except (TypeError, ValueError):
                errors.append("particle_size_nm must be numeric for composite rows.")

    return errors


# ---------------------------------------------------------------------------
# Dataset serialization helpers
# ---------------------------------------------------------------------------

def _serialize_dataset(ds: Dataset) -> dict[str, Any]:
    """Convert Dataset ORM object to response dict."""
    def _as_utc_iso(dt: datetime | None) -> str:
        if not dt:
            return ""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.isoformat()

    return {
        "id": str(ds.id),
        "display_name": ds.display_name,
        "original_filename": ds.original_filename,
        "status": ds.status,
        "total_rows": ds.total_rows,
        "total_columns": ds.total_columns,
        "column_mapping": ds.column_mapping or {},
        "has_composite_fields": ds.has_composite_fields,
        "uploaded_at": _as_utc_iso(ds.uploaded_at),
        "updated_at": _as_utc_iso(ds.updated_at),
    }


def _serialize_material(m: Material) -> dict[str, Any]:
    """Convert Material ORM object to response dict."""
    return {
        "id": str(m.id),
        "uid": m.uid,
        "formula": m.formula,
        "d33": m.d33,
        "tc": m.tc,
        "vickers_hardness": m.vickers_hardness,
        "qm": m.qm,
        "kp": m.kp,
        "relative_density_pct": m.relative_density_pct,
        "sintering_temp_c": m.sintering_temp_c,
        "sintering_method": m.sintering_method,
        "ceramic_type": m.ceramic_type,
        "fabrication_method": m.fabrication_method,
        "matrix_type": m.matrix_type,
        "filler_wt_pct": m.filler_wt_pct,
        "particle_morphology": m.particle_morphology,
        "particle_size_nm": m.particle_size_nm,
        "surface_treatment": m.surface_treatment,
        "source_doi": m.source_doi,
        "source_notes": m.source_notes,
        "parse_status": m.parse_status,
        "parse_warnings": m.parse_warnings,
        "source_row": m.source_row,
        "parsed_row": m.parsed_row,
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

async def upload_csv(file_content: bytes, filename: str, db: AsyncSession) -> UploadPreviewResponse:
    """
    Process an uploaded CSV file.

    1. Read CSV with pandas (handle encoding)
    2. Create Dataset record with status='pending'
    3. Save raw CSV to temp file
    4. Return preview with columns, row count, sample rows, suggested mapping
    """
    # Read CSV
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_content), encoding="latin-1")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")

    if df.empty:
        raise ValueError("CSV file is empty — no data rows found.")

    if len(df.columns) == 0:
        raise ValueError("CSV file has no columns.")

    # Create dataset record
    dataset_id = uuid.uuid4()
    display_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    columns = df.columns.tolist()

    dataset = Dataset(
        id=dataset_id,
        display_name=display_name,
        original_filename=filename,
        status="pending",
        total_rows=len(df),
        total_columns=len(columns),
        column_mapping={},
        has_composite_fields=False,
    )
    db.add(dataset)
    await db.flush()

    # Save raw CSV to temp file for later use during column mapping
    temp_path = _get_temp_csv_path(str(dataset_id))
    temp_path.write_bytes(file_content)

    # Auto-suggest column mapping
    suggested = suggest_column_mapping(columns)

    # Preview rows (first 10)
    preview_df = df.head(10).fillna("")
    preview_rows = preview_df.to_dict(orient="records")

    return UploadPreviewResponse(
        dataset_id=str(dataset_id),
        filename=filename,
        columns=columns,
        row_count=len(df),
        preview_rows=preview_rows,
        suggested_mapping=suggested,
    )


# ---------------------------------------------------------------------------
# Column Mapping
# ---------------------------------------------------------------------------

async def apply_column_mapping(
    dataset_id: str,
    mapping: dict[str, str],
    db: AsyncSession,
) -> DatasetDetailResponse:
    """
    Apply column mapping and save rows to materials table.

    1. Load raw CSV from temp file
    2. Rename columns per mapping (unmapped columns dropped)
    3. For each row: assign uid, apply bulk ceramic defaults, create Material
    4. Validate formulas via ml-core
    5. Update dataset metadata
    6. Clean up temp file
    """
    # Get dataset
    result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    # Read CSV from temp file
    temp_path = _get_temp_csv_path(dataset_id)
    if not temp_path.exists():
        raise ValueError(
            "Raw CSV file not found. Please re-upload the dataset."
        )

    try:
        df = pd.read_csv(temp_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(temp_path, encoding="latin-1")

    # Build reverse mapping: backend_field → csv_col
    reverse_map = {v: k for k, v in mapping.items()}

    # Select and rename only mapped columns
    mapped_cols = {}
    for backend_field, csv_col in reverse_map.items():
        if csv_col in df.columns:
            mapped_cols[csv_col] = backend_field

    if "formula" not in mapped_cols.values():
        raise ValueError("Column mapping must include 'formula'.")

    df_mapped = df[list(mapped_cols.keys())].rename(columns=mapped_cols)

    # Determine if dataset has composite fields
    has_composite = False
    if "filler_wt_pct" in df_mapped.columns:
        non_zero_filler = pd.to_numeric(df_mapped["filler_wt_pct"], errors="coerce").fillna(0)
        has_composite = (non_zero_filler > 0).any()
    if "matrix_type" in df_mapped.columns:
        non_none_matrix = df_mapped["matrix_type"].fillna("none").str.lower()
        has_composite = has_composite or (non_none_matrix != "none").any()

    # Delete any existing materials for this dataset (in case of re-mapping)
    await db.execute(
        delete(Material).where(Material.dataset_id == dataset.id)
    )

    # Create material rows with uid assignment
    formulas: list[str] = []
    materials: list[Material] = []

    for idx, row_data in enumerate(df_mapped.to_dict(orient="records"), start=1):
        # Apply bulk ceramic defaults
        row_data = apply_bulk_ceramic_defaults(row_data)

        # Clean numeric fields
        for num_field in [
            "d33", "tc", "vickers_hardness", "qm", "kp",
            "relative_density_pct", "sintering_temp_c",
            "filler_wt_pct", "particle_size_nm",
        ]:
            if num_field in row_data:
                val = row_data[num_field]
                if pd.isna(val) or val == "" or val is None:
                    row_data[num_field] = None
                else:
                    try:
                        row_data[num_field] = float(val)
                    except (ValueError, TypeError):
                        row_data[num_field] = None

        # Clean string fields
        for str_field in [
            "formula", "sintering_method", "ceramic_type", "fabrication_method",
            "matrix_type", "particle_morphology", "surface_treatment",
            "source_doi", "source_notes",
        ]:
            if str_field in row_data:
                val = row_data[str_field]
                if pd.isna(val) or val is None:
                    row_data[str_field] = None if str_field not in (
                        "matrix_type", "particle_morphology", "surface_treatment"
                    ) else "none"
                else:
                    row_data[str_field] = str(val).strip()

        # Ensure formula is not empty
        formula_val = row_data.get("formula", "")
        if not formula_val or (isinstance(formula_val, float) and pd.isna(formula_val)):
            formula_val = f"MISSING_ROW_{idx}"
        row_data["formula"] = str(formula_val)
        formulas.append(row_data["formula"])

        # Snapshot (source row) before normalization
        source_snapshot = dict(row_data)

        material = Material(
            dataset_id=dataset.id,
            uid=idx,
            **{k: v for k, v in row_data.items() if hasattr(Material, k)},
        )
        material.source_row = source_snapshot
        materials.append(material)

    # Validate formulas via ml-core
    validation_results = validate_formulas_batch(formulas)
    for material, vr in zip(materials, validation_results):
        material.parse_status = vr.parse_status
        material.parse_warnings = vr.parse_warnings_str
        # Parsed snapshot starts from source snapshot but reflects validator outputs + safe normalization
        parsed_snapshot = dict(material.source_row or {})
        parsed_snapshot["parse_status"] = vr.parse_status
        parsed_snapshot["parse_warnings"] = vr.parse_warnings_str
        parsed_snapshot["normalized_formula"] = vr.normalized_formula

        # Apply safe auto-fixes (normalized formula)
        if vr.normalized_formula != vr.formula and vr.normalized_formula:
            material.formula = vr.normalized_formula
            parsed_snapshot["formula"] = vr.normalized_formula
        else:
            parsed_snapshot["formula"] = material.formula

        material.parsed_row = parsed_snapshot

    # Bulk add materials
    db.add_all(materials)

    # Update dataset
    dataset.column_mapping = mapping
    dataset.total_rows = len(materials)
    dataset.total_columns = len(mapped_cols)
    dataset.has_composite_fields = has_composite
    dataset.status = "pending"  # Still pending until finalized

    await db.flush()

    # Clean up temp file
    _cleanup_temp_csv(dataset_id)

    return DatasetDetailResponse(**_serialize_dataset(dataset))


# ---------------------------------------------------------------------------
# Quality Report
# ---------------------------------------------------------------------------

async def get_quality_report(
    dataset_id: str,
    db: AsyncSession,
) -> DataQualityReport:
    """
    Generate data quality report for a dataset.

    Issues are generated ONLY for:
    - Missing required values (formula)
    - Unsupported elements (from parse_status)
    - Parse errors
    - Invalid categorical values

    Optional null fields (qm, kp, sintering_temp_c, etc.) are tracked
    in column_stats for informational display but do NOT create issues.
    """
    import logging
    logger = logging.getLogger("piezo.dataset")

    result = await db.execute(
        select(Material)
        .where(Material.dataset_id == uuid.UUID(dataset_id))
        .order_by(Material.uid)
    )
    materials = result.scalars().all()

    if not materials:
        raise ValueError(f"No materials found for dataset {dataset_id}")

    issues: list[DataIssue] = []
    valid_count = 0

    # Numeric fields to track stats for
    numeric_fields = [
        "d33", "tc", "vickers_hardness", "qm", "kp",
        "relative_density_pct", "sintering_temp_c",
        "filler_wt_pct", "particle_size_nm",
    ]

    # Track per-column stats
    col_stats_data: dict[str, dict[str, int]] = {}
    all_fields = ["formula"] + numeric_fields + [
        "sintering_method", "ceramic_type", "fabrication_method",
        "matrix_type", "particle_morphology", "surface_treatment",
    ]
    for f in all_fields:
        col_stats_data[f] = {"total": 0, "missing": 0, "invalid": 0}

    for m in materials:
        row_valid = True

        # ------ Parse status issues (formula validation) ------
        if m.parse_status == "error":
            issues.append(DataIssue(
                row_uid=m.uid,
                material_id=str(m.id),
                column="formula",
                issue_type="parse_error",
                current_value=m.formula,
                message=m.parse_warnings or "Formula could not be parsed",
            ))
            row_valid = False
            logger.debug("  uid=%d parse_error: %s", m.uid, m.formula)
        elif m.parse_status == "unsupported_elements":
            issues.append(DataIssue(
                row_uid=m.uid,
                material_id=str(m.id),
                column="formula",
                issue_type="unsupported_element",
                current_value=m.formula,
                message=m.parse_warnings or "Formula contains unsupported elements",
            ))
            row_valid = False
            logger.debug("  uid=%d unsupported_elements: %s", m.uid, m.formula)

        # ------ Check formula is not missing (REQUIRED field) ------
        col_stats_data["formula"]["total"] += 1
        if not m.formula or m.formula.startswith("MISSING_ROW_"):
            col_stats_data["formula"]["missing"] += 1
            # Only add missing_value issue if we didn't already add a parse issue
            # This prevents duplicate issues on the same material+column
            existing_formula_issue = any(
                i.material_id == str(m.id) and i.column == "formula"
                for i in issues
            )
            if not existing_formula_issue:
                issues.append(DataIssue(
                    row_uid=m.uid,
                    material_id=str(m.id),
                    column="formula",
                    issue_type="missing_value",
                    current_value=m.formula,
                    message="Formula is required but missing",
                ))
            row_valid = False

        # ------ Track numeric field stats (info-only, no issues for optional nulls) ------
        for nf in numeric_fields:
            val = getattr(m, nf, None)
            col_stats_data[nf]["total"] += 1
            if val is None:
                col_stats_data[nf]["missing"] += 1

        # ------ Check categorical fields for invalid values ------
        for cf in ["sintering_method", "ceramic_type", "fabrication_method"]:
            val = getattr(m, cf, None)
            col_stats_data[cf]["total"] += 1
            if val is None:
                col_stats_data[cf]["missing"] += 1
            elif cf in CATEGORICAL_FIELD_OPTIONS and val not in CATEGORICAL_FIELD_OPTIONS[cf]:
                col_stats_data[cf]["invalid"] += 1
                issues.append(DataIssue(
                    row_uid=m.uid,
                    material_id=str(m.id),
                    column=cf,
                    issue_type="invalid_character",
                    current_value=str(val),
                    message=f"Invalid value '{val}'. Expected: {', '.join(CATEGORICAL_FIELD_OPTIONS[cf])}",
                ))
                row_valid = False

        if row_valid:
            valid_count += 1

    # Build column stats — only include columns that have data or issues
    column_stats = []
    for field_name, stats in col_stats_data.items():
        if stats["total"] > 0 and (stats["missing"] > 0 or stats["invalid"] > 0):
            # Only show columns that have some missing or invalid values
            dtype = "float" if field_name in numeric_fields else "string"
            column_stats.append(ColumnStats(
                column_name=field_name,
                total_values=stats["total"],
                missing_count=stats["missing"],
                invalid_count=stats["invalid"],
                dtype=dtype,
            ))

    return DataQualityReport(
        total_rows=len(materials),
        valid_rows=valid_count,
        issue_count=len({i.material_id for i in issues}),  # unique rows with issues
        issues=issues,
        column_stats=column_stats,
    )


# ---------------------------------------------------------------------------
# Finalize Dataset
# ---------------------------------------------------------------------------

async def finalize_dataset(
    dataset_id: str,
    db: AsyncSession,
) -> DatasetDetailResponse:
    """Mark dataset as 'ready' after user resolves issues."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    dataset.status = "ready"
    await db.flush()
    return DatasetDetailResponse(**_serialize_dataset(dataset))


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

async def list_datasets(db: AsyncSession) -> list[DatasetSummaryResponse]:
    """List all datasets with summary info."""
    result = await db.execute(
        select(Dataset).order_by(Dataset.uploaded_at.desc())
    )
    datasets = result.scalars().all()
    return [DatasetSummaryResponse(**_serialize_dataset(ds)) for ds in datasets]


async def get_dataset(dataset_id: str, db: AsyncSession) -> DatasetDetailResponse:
    """Get full dataset detail."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    return DatasetDetailResponse(**_serialize_dataset(dataset))


async def rename_dataset(
    dataset_id: str,
    display_name: str,
    db: AsyncSession,
) -> DatasetDetailResponse:
    """Rename dataset (display_name only — UUID never changes)."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    dataset.display_name = display_name
    await db.flush()
    return DatasetDetailResponse(**_serialize_dataset(dataset))


async def delete_dataset(dataset_id: str, db: AsyncSession) -> None:
    """Delete dataset and all associated materials (cascade)."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    # Clean up temp file if it exists
    _cleanup_temp_csv(dataset_id)

    await db.delete(dataset)
    await db.flush()


# ---------------------------------------------------------------------------
# Material CRUD
# ---------------------------------------------------------------------------

async def get_materials(
    dataset_id: str,
    db: AsyncSession,
    search: str | None = None,
    sort_by: str = "uid",
    sort_order: str = "asc",
    page: int = 1,
    page_size: int = 100,
) -> PaginatedMaterialsResponse:
    """Get paginated materials for a dataset with search and sort."""
    base_query = select(Material).where(
        Material.dataset_id == uuid.UUID(dataset_id)
    )

    # Search filter — searches across all string fields + numeric cast
    if search:
        from sqlalchemy import or_, cast, String
        search_clauses = [
            Material.formula.ilike(f"%{search}%"),
            Material.sintering_method.ilike(f"%{search}%"),
            Material.ceramic_type.ilike(f"%{search}%"),
            Material.fabrication_method.ilike(f"%{search}%"),
            Material.matrix_type.ilike(f"%{search}%"),
            Material.particle_morphology.ilike(f"%{search}%"),
            Material.surface_treatment.ilike(f"%{search}%"),
            Material.parse_status.ilike(f"%{search}%"),
            Material.source_doi.ilike(f"%{search}%"),
            Material.source_notes.ilike(f"%{search}%"),
            cast(Material.uid, String).ilike(f"%{search}%"),
            cast(Material.d33, String).ilike(f"%{search}%"),
            cast(Material.tc, String).ilike(f"%{search}%"),
            cast(Material.vickers_hardness, String).ilike(f"%{search}%"),
        ]
        base_query = base_query.where(or_(*search_clauses))

    # Count total
    count_query = select(func.count()).select_from(
        base_query.subquery()
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Sort
    sort_col = getattr(Material, sort_by, Material.uid)
    if sort_order == "desc":
        base_query = base_query.order_by(sort_col.desc())
    else:
        base_query = base_query.order_by(sort_col.asc())

    # Paginate
    offset = (page - 1) * page_size
    base_query = base_query.offset(offset).limit(page_size)

    result = await db.execute(base_query)
    materials = result.scalars().all()

    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedMaterialsResponse(
        items=[MaterialResponse(**_serialize_material(m)) for m in materials],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


async def add_material(
    dataset_id: str,
    data: dict[str, Any],
    db: AsyncSession,
) -> MaterialResponse:
    """Add a new material row to a dataset with next available uid."""
    # Get next uid
    result = await db.execute(
        select(func.max(Material.uid)).where(
            Material.dataset_id == uuid.UUID(dataset_id)
        )
    )
    max_uid = result.scalar() or 0
    next_uid = max_uid + 1

    # Validate consistency before any defaults
    consistency_errors = validate_material_consistency(data)
    if consistency_errors:
        raise ValueError(" | ".join(consistency_errors))

    # Apply bulk ceramic defaults for strict bulk rows
    data = apply_bulk_ceramic_defaults(data)

    # Validate formula
    from piezo_ml.validators.formula_validator import validate_formula
    vr = validate_formula(data.get("formula", ""))

    material = Material(
        dataset_id=uuid.UUID(dataset_id),
        uid=next_uid,
        parse_status=vr.parse_status,
        parse_warnings=vr.parse_warnings_str,
        **{k: v for k, v in data.items() if hasattr(Material, k) and k not in ("parse_status", "parse_warnings")},
    )
    material.source_row = {k: v for k, v in _serialize_material(material).items() if k not in ("source_row", "parsed_row")}
    material.parsed_row = {
        **(material.source_row or {}),
        "parse_status": vr.parse_status,
        "parse_warnings": vr.parse_warnings_str,
        "normalized_formula": vr.normalized_formula,
        "formula": vr.normalized_formula if vr.normalized_formula else material.formula,
    }
    if vr.normalized_formula != vr.formula:
        material.formula = vr.normalized_formula

    db.add(material)

    # Update dataset row count
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset:
        dataset.total_rows = next_uid
        if dataset.status == "ready":
            dataset.status = "pending"

    await db.flush()
    return MaterialResponse(**_serialize_material(material))


async def update_material(
    dataset_id: str,
    material_id: str,
    data: dict[str, Any],
    db: AsyncSession,
) -> MaterialResponse:
    """Update a single material row."""
    result = await db.execute(
        select(Material).where(
            Material.id == uuid.UUID(material_id),
            Material.dataset_id == uuid.UUID(dataset_id),
        )
    )
    material = result.scalar_one_or_none()
    if not material:
        raise ValueError(f"Material {material_id} not found in dataset {dataset_id}")

    # Validate and apply updates
    candidate = _serialize_material(material)
    candidate.update(data)

    consistency_errors = validate_material_consistency(candidate)
    if consistency_errors:
        raise ValueError(" | ".join(consistency_errors))

    for field_name, value in data.items():
        if hasattr(material, field_name) and field_name not in ("id", "uid", "dataset_id"):
            # Basic type validation
            if field_name in (
                "d33", "tc", "vickers_hardness", "qm", "kp",
                "relative_density_pct", "sintering_temp_c",
                "filler_wt_pct", "particle_size_nm",
            ):
                if value is not None and not isinstance(value, (int, float)):
                    raise ValueError(f"Field '{field_name}' must be a number or null")
            if field_name in CATEGORICAL_FIELD_OPTIONS and value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"Field '{field_name}' must be a string or null")
                if value not in CATEGORICAL_FIELD_OPTIONS[field_name]:
                    raise ValueError(
                        f"Invalid value '{value}' for '{field_name}'. "
                        f"Expected: {', '.join(CATEGORICAL_FIELD_OPTIONS[field_name])}"
                    )
            setattr(material, field_name, value)

    # Re-validate formula if changed — validate the NEW value, not the DB value
    if "formula" in data:
        from piezo_ml.validators.formula_validator import validate_formula
        new_formula = data["formula"]
        vr = validate_formula(new_formula)
        material.parse_status = vr.parse_status
        material.parse_warnings = vr.parse_warnings_str
        material.formula = vr.normalized_formula  # always update to normalized

    # Refresh snapshots after any change
    material.source_row = {k: v for k, v in _serialize_material(material).items() if k not in ("source_row", "parsed_row")}
    material.parsed_row = {
        **(material.source_row or {}),
        "parse_status": material.parse_status,
        "parse_warnings": material.parse_warnings,
        "normalized_formula": material.formula,
        "formula": material.formula,
    }

    # Re-apply bulk ceramic defaults if composite fields changed
    if any(f in data for f in ("filler_wt_pct", "matrix_type")):
        row_dict = _serialize_material(material)
        row_dict = apply_bulk_ceramic_defaults(row_dict)
        for k in ("matrix_type", "particle_morphology", "surface_treatment", "particle_size_nm", "filler_wt_pct"):
            setattr(material, k, row_dict[k])

    await db.flush()

    # Mark dataset pending after modifications (forces re-validation/finalize)
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset and dataset.status == "ready":
        dataset.status = "pending"
        await db.flush()

    return MaterialResponse(**_serialize_material(material))


async def bulk_update_materials(
    dataset_id: str,
    updates: list[dict[str, Any]],
    deletes: list[str],
    db: AsyncSession,
) -> BulkUpdateResponse:
    """Bulk update and/or delete materials."""
    errors: list[str] = []
    updated_count = 0
    deleted_count = 0

    # Process updates
    for item in updates:
        mid = item.get("id")
        update_data = item.get("updates", {})
        material = None
        try:
            result = await db.execute(
                select(Material).where(
                    Material.id == uuid.UUID(mid),
                    Material.dataset_id == uuid.UUID(dataset_id),
                )
            )
            material = result.scalar_one_or_none()
            if not material:
                errors.append(f"Update {mid}: not found")
                continue
            await update_material(dataset_id, mid, update_data, db)
            updated_count += 1
        except Exception as e:
            uid_txt = f"uid={getattr(material, 'uid', '?')} " if "material" in locals() and material else ""
            errors.append(f"Update {uid_txt}({mid}): {str(e)}")

    # Process deletes
    for mid in deletes:
        material = None
        try:
            result = await db.execute(
                select(Material).where(
                    Material.id == uuid.UUID(mid),
                    Material.dataset_id == uuid.UUID(dataset_id),
                )
            )
            material = result.scalar_one_or_none()
            if material:
                await db.delete(material)
                deleted_count += 1
            else:
                errors.append(f"Delete {mid}: not found")
        except Exception as e:
            uid_txt = f"uid={getattr(material, 'uid', '?')} " if "material" in locals() and material else ""
            errors.append(f"Delete {uid_txt}({mid}): {str(e)}")

    # Update dataset row count
    count_result = await db.execute(
        select(func.count()).where(Material.dataset_id == uuid.UUID(dataset_id))
    )
    new_count = count_result.scalar() or 0
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset:
        dataset.total_rows = new_count
        if dataset.status == "ready" and (updated_count > 0 or deleted_count > 0):
            dataset.status = "pending"

    await db.flush()

    return BulkUpdateResponse(
        updated_count=updated_count,
        deleted_count=deleted_count,
        errors=errors,
    )


async def delete_material(
    dataset_id: str,
    material_id: str,
    db: AsyncSession,
) -> None:
    """Delete a single material row."""
    result = await db.execute(
        select(Material).where(
            Material.id == uuid.UUID(material_id),
            Material.dataset_id == uuid.UUID(dataset_id),
        )
    )
    material = result.scalar_one_or_none()
    if not material:
        raise ValueError(f"Material {material_id} not found")

    await db.delete(material)

    # Update dataset row count
    count_result = await db.execute(
        select(func.count()).where(Material.dataset_id == uuid.UUID(dataset_id))
    )
    new_count = count_result.scalar() or 0
    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset:
        dataset.total_rows = new_count
        if dataset.status == "ready":
            dataset.status = "pending"

    await db.flush()


# ---------------------------------------------------------------------------
# Copy dataset
# ---------------------------------------------------------------------------

async def copy_dataset(
    dataset_id: str,
    new_name: str | None,
    db: AsyncSession,
) -> DatasetDetailResponse:
    """Copy a dataset and all its materials. UUID is new, display_name customizable."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    source_ds = result.scalar_one_or_none()
    if not source_ds:
        raise ValueError(f"Dataset {dataset_id} not found")

    display = new_name or f"{source_ds.display_name} (copy)"
    new_ds = Dataset(
        display_name=display,
        original_filename=source_ds.original_filename,
        status="pending",
        total_rows=source_ds.total_rows,
        total_columns=source_ds.total_columns,
        column_mapping=source_ds.column_mapping,
        has_composite_fields=source_ds.has_composite_fields,
    )
    db.add(new_ds)
    await db.flush()

    # Copy materials
    mat_result = await db.execute(
        select(Material).where(Material.dataset_id == uuid.UUID(dataset_id)).order_by(Material.uid)
    )
    source_materials = mat_result.scalars().all()

    for m in source_materials:
        new_mat = Material(
            dataset_id=new_ds.id,
            uid=m.uid,
            formula=m.formula, d33=m.d33, tc=m.tc,
            vickers_hardness=m.vickers_hardness, qm=m.qm, kp=m.kp,
            relative_density_pct=m.relative_density_pct,
            sintering_temp_c=m.sintering_temp_c,
            sintering_method=m.sintering_method,
            ceramic_type=m.ceramic_type,
            fabrication_method=m.fabrication_method,
            matrix_type=m.matrix_type,
            filler_wt_pct=m.filler_wt_pct,
            particle_morphology=m.particle_morphology,
            particle_size_nm=m.particle_size_nm,
            surface_treatment=m.surface_treatment,
            source_doi=m.source_doi, source_notes=m.source_notes,
            parse_status=m.parse_status, parse_warnings=m.parse_warnings,
            source_row=m.source_row, parsed_row=m.parsed_row,
        )
        db.add(new_mat)

    await db.flush()
    return DatasetDetailResponse(**_serialize_dataset(new_ds))


# ---------------------------------------------------------------------------
# Bulk delete datasets
# ---------------------------------------------------------------------------

async def bulk_delete_datasets(
    dataset_ids: list[str],
    db: AsyncSession,
) -> dict[str, Any]:
    """Delete multiple datasets at once. Returns count of deleted."""
    deleted = 0
    errors: list[str] = []
    for did in dataset_ids:
        try:
            await delete_dataset(did, db)
            deleted += 1
        except ValueError as e:
            errors.append(str(e))
    return {"deleted_count": deleted, "errors": errors}


# ---------------------------------------------------------------------------
# Column clear (Review Issues remediation)
# ---------------------------------------------------------------------------

async def clear_column(
    dataset_id: str,
    field_name: str,
    db: AsyncSession,
) -> BulkUpdateResponse:
    """
    Clear a specific field across all materials in a dataset.
    - For numeric/string nullable fields: set to NULL
    - For composite sentinel fields: reset to defaults if clearing would violate constraints
    Marks dataset as pending if it was ready.
    """
    allowed = {
        "formula",
        "d33",
        "tc",
        "vickers_hardness",
        "qm",
        "kp",
        "relative_density_pct",
        "sintering_temp_c",
        "sintering_method",
        "ceramic_type",
        "fabrication_method",
        "matrix_type",
        "filler_wt_pct",
        "particle_morphology",
        "particle_size_nm",
        "surface_treatment",
        "source_doi",
        "source_notes",
    }
    if field_name not in allowed:
        raise ValueError(f"Field '{field_name}' cannot be cleared.")

    result = await db.execute(
        select(Material).where(Material.dataset_id == uuid.UUID(dataset_id))
    )
    materials = result.scalars().all()
    if not materials:
        return BulkUpdateResponse(updated_count=0, deleted_count=0, errors=[])

    updated = 0
    errors: list[str] = []

    for m in materials:
        try:
            if field_name in ("matrix_type", "particle_morphology", "surface_treatment"):
                setattr(m, field_name, "none")
            elif field_name == "filler_wt_pct":
                setattr(m, field_name, 0)
            else:
                setattr(m, field_name, None)

            # Re-apply defaults and re-validate formula if needed
            row_dict = _serialize_material(m)
            row_dict = apply_bulk_ceramic_defaults(row_dict)
            for k in ("matrix_type", "particle_morphology", "surface_treatment", "particle_size_nm", "filler_wt_pct"):
                setattr(m, k, row_dict[k])

            if field_name == "formula":
                from piezo_ml.validators.formula_validator import validate_formula
                vr = validate_formula(m.formula or "")
                m.parse_status = vr.parse_status
                m.parse_warnings = vr.parse_warnings_str
                if vr.normalized_formula:
                    m.formula = vr.normalized_formula

            m.source_row = {k: v for k, v in _serialize_material(m).items() if k not in ("source_row", "parsed_row")}
            m.parsed_row = {
                **(m.source_row or {}),
                "parse_status": m.parse_status,
                "parse_warnings": m.parse_warnings,
                "normalized_formula": m.formula,
                "formula": m.formula,
            }
            updated += 1
        except Exception as e:
            errors.append(f"uid={getattr(m, 'uid', '?')}: {str(e)}")

    ds_result = await db.execute(
        select(Dataset).where(Dataset.id == uuid.UUID(dataset_id))
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset and dataset.status == "ready" and updated > 0:
        dataset.status = "pending"

    await db.flush()
    return BulkUpdateResponse(updated_count=updated, deleted_count=0, errors=errors)
