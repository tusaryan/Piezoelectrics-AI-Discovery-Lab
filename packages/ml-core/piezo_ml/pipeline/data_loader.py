import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from io import BytesIO

from piezo_ml.pipeline.data_validator import DataValidator, DataIssue
from piezo_ml.pipeline.data_cleaner import DataCleaner
from piezo_ml.features.engineer import FeatureEngineer

# Known aliases for internal fields — used for fuzzy column matching
COLUMN_ALIASES: Dict[str, List[str]] = {
    "formula": ["formula", "composition", "chemical_formula", "chem_formula", "material", "compound", "chemical composition"],
    "d33": ["d33", "d_33", "piezoelectric_coefficient", "piezo_coeff", "d33_pc_n", "d33 (pc/n)"],
    "tc": ["tc", "t_c", "curie_temp", "curie_temperature", "curie temperature", "tc (°c)", "tc_c", "curie"],
    "sintering_temp": ["sintering_temp", "sintering_temperature", "sintering temperature", "sinter_temp", "ts", "sintering temp"],
    "family_name": ["family_name", "family", "material_family", "type", "category"],
    "field_strength": ["field_strength", "field strength", "electric_field", "e_field"],
    "poling_temp": ["poling_temp", "poling_temperature", "poling temperature"],
    "poling_time": ["poling_time", "poling time"],
    "density": ["density", "rho", "ρ"],
    "density_theoretical_pct": ["density_theoretical_pct", "density_pct", "theoretical_density", "relative_density", "relative density"],
    "planar_coupling": ["planar_coupling", "kp", "k_p", "planar coupling"],
    "dielectric_const": ["dielectric_const", "dielectric_constant", "epsilon", "εr", "permittivity", "dielectric constant"],
    "dielectric_loss": ["dielectric_loss", "tan_delta", "tanδ", "loss_tangent", "dielectric loss"],
    "mech_quality_factor": ["mech_quality_factor", "qm", "q_m", "mechanical_quality", "mechanical quality factor"],
}

# Internal fields that the schema supports
INTERNAL_FIELDS = list(COLUMN_ALIASES.keys())

class DataLoader:
    """
    Loads raw CSV/XLSX or DataFrames, orchestrating validation and structuring.
    """
    
    @staticmethod
    def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
        if filename.endswith('.csv'):
            return pd.read_csv(BytesIO(file_bytes))
        elif filename.endswith('.xlsx'):
            return pd.read_excel(BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def suggest_column_mapping(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Fuzzy-match CSV columns against internal schema fields.
        Returns {internal_field: {"csv_column": str|None, "confidence": float}}
        """
        csv_cols = list(df.columns)
        csv_cols_lower = [c.lower().strip() for c in csv_cols]
        
        mapping: Dict[str, Dict[str, Any]] = {}
        used_csv_cols: set = set()
        
        for internal_field, aliases in COLUMN_ALIASES.items():
            best_match = None
            best_confidence = 0.0
            
            for i, csv_lower in enumerate(csv_cols_lower):
                if csv_cols[i] in used_csv_cols:
                    continue
                    
                # Exact match (case-insensitive)
                if csv_lower in [a.lower() for a in aliases]:
                    best_match = csv_cols[i]
                    best_confidence = 1.0
                    break
                    
                # Substring match — csv col contains alias or vice versa
                for alias in aliases:
                    alias_lower = alias.lower()
                    if alias_lower in csv_lower or csv_lower in alias_lower:
                        conf = len(min(alias_lower, csv_lower, key=len)) / len(max(alias_lower, csv_lower, key=len))
                        if conf > best_confidence:
                            best_confidence = conf
                            best_match = csv_cols[i]
            
            mapping[internal_field] = {
                "csv_column": best_match,
                "confidence": round(best_confidence, 2)
            }
            if best_match:
                used_csv_cols.add(best_match)
        
        return mapping

    @staticmethod
    def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Rename CSV columns based on user-confirmed mapping.
        mapping: {internal_field: csv_column_name}
        Only renames columns that are present; ignores unmapped columns.
        """
        rename_dict = {}
        for internal_field, csv_col in mapping.items():
            if csv_col and csv_col in df.columns and csv_col != internal_field:
                rename_dict[csv_col] = internal_field
        
        return df.rename(columns=rename_dict)

    @staticmethod
    def inspect(df: pd.DataFrame) -> Tuple[List[DataIssue], Dict[str, Any]]:
        """
        Inspect dataset before fully committing/cleaning it. 
        Returns issues and schema metadata.
        """
        issues = DataValidator.validate_df(df)
        
        metadata = {
            'row_count': len(df),
            'columns': list(df.columns),
            'has_d33': 'd33' in df.columns,
            'has_tc': 'tc' in df.columns,
            'issue_count': len(issues),
            'critical_issue_count': sum(1 for i in issues if i.severity == 'critical')
        }
        return issues, metadata
        
    @staticmethod
    def process_and_extract(df: pd.DataFrame, resolutions: Dict[int, str], progress_callback=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Apply resolutions -> Extract Features -> Returns (X, y_d33, y_tc, metadata_df)
        """
        clean_df = DataCleaner.apply_fixes(df, resolutions)
        
        engineer = FeatureEngineer()
        
        X_list = []
        valid_indices = []
        
        total = len(clean_df)
        for idx, (row_idx, row) in enumerate(clean_df.iterrows()):
            try:
                vec, _ = engineer.compute_features(row['formula'])
                X_list.append(vec)
                valid_indices.append(row_idx)
            except Exception:
                pass
            
            # Report progress between 10% and 90%
            if progress_callback and total > 0 and idx % max(1, total // 100) == 0:
                progress_callback(10 + int((idx / total) * 80))
                
        final_df = clean_df.loc[valid_indices].reset_index(drop=True)
        
        X = np.array(X_list)
        y_d33 = final_df['d33'].values if 'd33' in final_df.columns else np.zeros(len(X))
        y_tc = final_df['tc'].values if 'tc' in final_df.columns else np.zeros(len(X))
        
        return X, y_d33, y_tc, final_df

