import uuid
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from packages.db.models.dataset import Dataset, Material
from piezo_ml.pipeline.data_loader import DataLoader
from piezo_ml.pipeline.data_validator import DataIssue

# Global progress tracker for SSE streaming
# format: { "dataset_id": progress_percentage }
DATASET_PROGRESS: Dict[str, Any] = {}

class DatasetService:
    @staticmethod
    async def process_upload_step1(db: AsyncSession, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Step 1: Parse file, suggest column mapping, save raw CSV.
        Returns dataset_id, csv_columns, suggested_mapping for the frontend SchemaMapper.
        """
        df = DataLoader.load_file(file_bytes, filename)
        suggested_mapping = DataLoader.suggest_column_mapping(df)
        
        dataset_id = str(uuid.uuid4())
        dataset = Dataset(
            id=dataset_id,
            name=filename,
            status="pending_mapping",
            row_count=len(df),
            has_d33=False,
            has_tc=False
        )
        db.add(dataset)
        
        from apps.api.app.core.config import settings
        import os
        raw_path = os.path.join(settings.model_artifacts_path, f"dataset_{dataset_id}_raw.csv")
        df.to_csv(raw_path, index=False)
        
        await db.commit()
        await db.refresh(dataset)
        
        return {
            "dataset_id": str(dataset.id),
            "csv_columns": list(df.columns),
            "row_count": len(df),
            "suggested_mapping": suggested_mapping
        }

    @staticmethod
    async def process_upload_step2(db: AsyncSession, dataset_id: str, column_mapping: Dict[str, str]) -> Tuple[Dataset, List[DataIssue]]:
        """
        Step 2: Apply user-confirmed column mapping, validate, insert materials or return issues.
        column_mapping: {internal_field: csv_column_name} e.g. {"formula": "Formula", "d33": "d33"}
        """
        from apps.api.app.core.config import settings
        import os
        raw_path = os.path.join(settings.model_artifacts_path, f"dataset_{dataset_id}_raw.csv")
        
        if not os.path.exists(raw_path):
            raise ValueError(f"Raw dataset file not found for {dataset_id}")
        
        df = pd.read_csv(raw_path)
        
        # Apply the user-confirmed column mapping
        df = DataLoader.apply_column_mapping(df, column_mapping)
        
        # Save the remapped version back
        df.to_csv(raw_path, index=False)
        
        # Now validate with mapped columns
        issues, metadata = DataLoader.inspect(df)
        
        dataset = await DatasetService.get_dataset(db, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset.has_d33 = metadata.get('has_d33', False)
        dataset.has_tc = metadata.get('has_tc', False)
        
        if issues:
            dataset.status = "pending_resolution"
            await db.commit()
            await db.refresh(dataset)
            return dataset, issues
        else:
            # No issues — process directly in background
            dataset.status = "processing"
            await db.commit()
            await db.refresh(dataset)
            # Cannot use background_tasks here if it's not passed, but we will pass it from router
            return dataset, issues

    @staticmethod
    async def _run_processing_task(dataset_id: str, resolutions: Dict[int, str] = None):
        import asyncio
        from apps.api.app.core.database import AsyncSessionLocal
        from apps.api.app.core.config import settings
        import os

        if resolutions is None:
            resolutions = {}

        def _update_progress(percent: int):
            DATASET_PROGRESS[dataset_id] = percent

        async with AsyncSessionLocal() as db:
            try:
                DATASET_PROGRESS[dataset_id] = 5
                raw_path = os.path.join(settings.model_artifacts_path, f"dataset_{dataset_id}_raw.csv")
                df = pd.read_csv(raw_path)
                
                # CPU-bound Pandas/Matminer processing runs OUTSIDE the main async loop!
                X, y_d33, y_tc, clean_df = await asyncio.to_thread(DataLoader.process_and_extract, df, resolutions, _update_progress)
                DATASET_PROGRESS[dataset_id] = 95
                
                dataset = await DatasetService.get_dataset(db, dataset_id)
                if not dataset:
                    return

                # Insert to DB
                for idx, row in clean_df.iterrows():
                    mat = Material(
                        dataset_id=dataset_id,
                        formula=str(row['formula']),
                        # CORE
                        d33=row.get('d33') if pd.notna(row.get('d33')) else None,
                        tc=row.get('tc') if pd.notna(row.get('tc')) else None,
                        is_imputed=bool(row.get('is_imputed', False)),
                        is_tc_ai_generated=bool(row.get('is_tc_ai_generated', False)),
                        # EXTENDED KNN BULK
                        family_name=row.get('family_name') if pd.notna(row.get('family_name')) else None,
                        sintering_temp=row.get('sintering_temp') if pd.notna(row.get('sintering_temp')) else None,
                        field_strength=row.get('field_strength') if pd.notna(row.get('field_strength')) else None,
                        poling_temp=row.get('poling_temp') if pd.notna(row.get('poling_temp')) else None,
                        poling_time=row.get('poling_time') if pd.notna(row.get('poling_time')) else None,
                        density=row.get('density') if pd.notna(row.get('density')) else None,
                        density_theoretical_pct=row.get('density_theoretical_pct') if pd.notna(row.get('density_theoretical_pct')) else None,
                        planar_coupling=row.get('planar_coupling') if pd.notna(row.get('planar_coupling')) else None,
                        dielectric_const=row.get('dielectric_const') if pd.notna(row.get('dielectric_const')) else None,
                        dielectric_loss=row.get('dielectric_loss') if pd.notna(row.get('dielectric_loss')) else None,
                        mech_quality_factor=row.get('mech_quality_factor') if pd.notna(row.get('mech_quality_factor')) else None,
                        # HARDNESS
                        vickers_hardness=row.get('vickers_hardness') if pd.notna(row.get('vickers_hardness')) else None,
                        mohs_hardness=row.get('mohs_hardness') if pd.notna(row.get('mohs_hardness')) else None,
                        # COMPOSITE
                        is_composite=bool(pd.notna(row.get('matrix_type')) or bool(row.get('is_composite', False))),
                        matrix_type=row.get('matrix_type') if pd.notna(row.get('matrix_type')) else None,
                        filler_wt_pct=row.get('filler_wt_pct') if pd.notna(row.get('filler_wt_pct')) else None,
                        filler_vol_pct=row.get('filler_vol_pct') if pd.notna(row.get('filler_vol_pct')) else None,
                        particle_morphology=row.get('particle_morphology') if pd.notna(row.get('particle_morphology')) else None,
                        particle_size_nm=row.get('particle_size_nm') if pd.notna(row.get('particle_size_nm')) else None,
                        surface_treatment=row.get('surface_treatment') if pd.notna(row.get('surface_treatment')) else None,
                        fabrication_method=row.get('fabrication_method') if pd.notna(row.get('fabrication_method')) else None,
                        beta_phase_pct=row.get('beta_phase_pct') if pd.notna(row.get('beta_phase_pct')) else None,
                        composite_d33=row.get('composite_d33') if pd.notna(row.get('composite_d33')) else None,
                        remnant_polarization=row.get('remnant_polarization') if pd.notna(row.get('remnant_polarization')) else None,
                        coercive_field=row.get('coercive_field') if pd.notna(row.get('coercive_field')) else None,
                    )
                    db.add(mat)
                
                dataset.status = "ready"
                dataset.row_count = len(clean_df)
                await db.commit()
                
                # Index to knowledge base
                try:
                    from piezo_ml.rag.knowledge_base import KnowledgeBase
                    kb = KnowledgeBase(settings.chroma_persist_path)
                    result = await db.execute(select(Material).where(Material.dataset_id == dataset_id))
                    materials = result.scalars().all()
                    for mat in materials:
                        material_dict = {
                            "id": str(mat.id),
                            "formula": getattr(mat, "formula", "unknown"),
                            "d33": getattr(mat, "d33", None),
                            "tc": getattr(mat, "tc", None),
                            "family_name": getattr(mat, "family_name", "N/A"),
                            "notes": getattr(mat, "notes", "")
                        }
                        kb.index_material(material_dict)
                except Exception:
                    pass
                    
                # Signal frontend that processing is complete
                DATASET_PROGRESS[dataset_id] = 100

            except Exception as e:
                import traceback
                print(f"Background ML processing failed: {e}\n{traceback.format_exc()}")
                DATASET_PROGRESS[dataset_id] = f"Error: {str(e)}"
                
                dataset = await DatasetService.get_dataset(db, dataset_id)
                if dataset:
                    dataset.status = "error"
                    await db.commit()

    @staticmethod
    async def list_datasets(db: AsyncSession) -> List[Dataset]:
        result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def get_dataset(db: AsyncSession, dataset_id: str) -> Optional[Dataset]:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        return result.scalars().first()
        
    @staticmethod
    async def get_materials(db: AsyncSession, dataset_id: str, skip: int = 0, limit: int = 50) -> Tuple[List[Material], int]:
        count_q = await db.execute(select(func.count()).where(Material.dataset_id == dataset_id))
        total = count_q.scalar() or 0
        
        result = await db.execute(
            select(Material)
            .where(Material.dataset_id == dataset_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all()), total
        
    @staticmethod
    async def update_material(db: AsyncSession, material_id: str, updates: Dict[str, Any]) -> Optional[Material]:
        result = await db.execute(select(Material).where(Material.id == material_id))
        mat = result.scalars().first()
        if not mat:
            return None
            
        for k, v in updates.items():
            if v is not None:
                setattr(mat, k, v)
                
        await db.commit()
        await db.refresh(mat)
        return mat

    @staticmethod
    async def get_dataset_issues(dataset_id: str) -> List[DataIssue]:
        from apps.api.app.core.config import settings
        import os
        raw_path = os.path.join(settings.model_artifacts_path, f"dataset_{dataset_id}_raw.csv")
        if not os.path.exists(raw_path):
            return []
        
        df = pd.read_csv(raw_path)
        issues, _ = DataLoader.inspect(df)
        return issues
        
    @staticmethod
    async def resolve_dataset_issues(db: AsyncSession, dataset_id: str, resolutions: Dict[int, str]) -> Tuple[bool, List[DataIssue], Optional[Dataset]]:
        from apps.api.app.core.config import settings
        from piezo_ml.pipeline.data_cleaner import DataCleaner
        import os
        raw_path = os.path.join(settings.model_artifacts_path, f"dataset_{dataset_id}_raw.csv")
        
        if not os.path.exists(raw_path):
            return False, [], None
            
        df = pd.read_csv(raw_path)
        
        # Fast clean just to check for remaining issues without ML feature extraction
        clean_df = DataCleaner.apply_fixes(df, resolutions)
        
        # Check if issues remain
        remaining_issues, _ = DataLoader.inspect(clean_df)
        if remaining_issues:
            clean_df.to_csv(raw_path, index=False)
            return False, remaining_issues, None
            
        dataset = await DatasetService.get_dataset(db, dataset_id)
        if not dataset:
            return False, [], None
            
        # All resolved! We save the cleaned data back so the background task can process it with empty resolutions
        clean_df.to_csv(raw_path, index=False)
        
        dataset.status = "processing"
        await db.commit()
        await db.refresh(dataset)
        
        return True, [], dataset

    @staticmethod
    async def delete_dataset(db: AsyncSession, dataset_id: str) -> bool:
        """Delete a dataset and all its materials. Also removes the raw CSV from disk."""
        dataset = await DatasetService.get_dataset(db, dataset_id)
        if not dataset:
            return False

        from sqlalchemy import delete as sa_delete
        await db.execute(sa_delete(Material).where(Material.dataset_id == dataset_id))
        await db.delete(dataset)
        await db.commit()

        try:
            from apps.api.app.core.config import settings
            import os
            raw_path = os.path.join(settings.model_artifacts_path, f"dataset_{dataset_id}_raw.csv")
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass

        DATASET_PROGRESS.pop(dataset_id, None)
        return True

