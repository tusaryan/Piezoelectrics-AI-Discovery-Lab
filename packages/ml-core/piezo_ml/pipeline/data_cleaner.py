from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from piezo_ml.features.engineer import FeatureEngineer

class DataCleaner:
    """
    Applies fixes (including automatic KNN Imputation) to a dataset.
    Supports inline edit, keep, auto-fix, and KNN impute resolutions.
    """
    
    # Unicode subscript/superscript replacements for auto-fix
    # Must match DataValidator.UNICODE_FIXES exactly
    UNICODE_FIXES = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '–': '-', '—': '-', '−': '-',
        '·': '.',
        '（': '(', '）': ')',
        '［': '[', '］': ']',
    }
    
    @classmethod
    def apply_fixes(cls, df: pd.DataFrame, resolutions: Dict[str, str]) -> pd.DataFrame:
        """
        resolutions mapping: {"row_idx::column::issue_type": choice_string}
        Supported choices:
          - "Drop Row" — remove the row
          - "KNN Impute" — fill missing values via KNN
          - "Keep Empty" / "Keep (suspicious)" / "Keep As-Is" / "Proceed Anyway" — no-op
          - "Auto-Fix" — apply Unicode normalization to formula
          - "edit:<value>" — replace the problematic cell with the user-provided value
          - "Go Back to Mapping" — no-op (handled by frontend)
        """
        clean_df = df.copy()
        
        # Auto-fixes universally applied
        if 'formula' in clean_df.columns:
            clean_df['formula'] = clean_df['formula'].astype(str).str.strip()
            
        rows_to_drop = set()
        impute_d33_rows = set()
        impute_tc_rows = set()
        
        # Process each resolution
        for comp_key, choice in resolutions.items():
            parts = str(comp_key).split("::")
            if len(parts) >= 2:
                try: idx = int(parts[0])
                except: continue
                col = parts[1]
            else:
                try: idx = int(comp_key)
                except: continue
                col = 'formula' # legacy fallback
                
            if idx < 0 or idx not in clean_df.index:
                continue  # Global issues or already dropped
            
            if choice == "Drop Row":
                rows_to_drop.add(idx)
            elif choice == "KNN Impute":
                if col == "d33": impute_d33_rows.add(idx)
                elif col == "tc": impute_tc_rows.add(idx)
            elif choice.startswith('edit:'):
                # Inline edit — determine which column based on the issue
                edited_value = choice[5:]
                if col in clean_df.columns:
                    clean_df.at[idx, col] = edited_value
            elif choice == 'Auto-Fix' and col == 'formula' and 'formula' in clean_df.columns:
                formula = str(clean_df.at[idx, 'formula'])
                for old, new in cls.UNICODE_FIXES.items():
                    formula = formula.replace(old, new)
                clean_df.at[idx, 'formula'] = formula.strip()
            
        # Drop rows
        if rows_to_drop:
            clean_df = clean_df.drop(index=list(rows_to_drop))
            
        # KNN Imputation
        all_impute = list(impute_d33_rows.union(impute_tc_rows))
        all_impute = [i for i in all_impute if i in clean_df.index]
        if all_impute and ('d33' in clean_df.columns or 'tc' in clean_df.columns):
            clean_df = cls._knn_impute(clean_df, all_impute)
            
        return clean_df.reset_index(drop=True)

    @classmethod
    def _knn_impute(cls, df: pd.DataFrame, target_rows: List[int], n_neighbors: int = 5) -> pd.DataFrame:
        engineer = FeatureEngineer()
        
        # Build feature matrix for KNN
        valid_indices = []
        vectors = []
        
        for idx in df.index:
            formula = str(df.loc[idx, 'formula'])
            try:
                vec, _ = engineer.compute_features(formula)
                vectors.append(vec)
                valid_indices.append(idx)
            except:
                pass
                
        if not vectors:
            return df
            
        X = np.array(vectors)
        
        # Build composite matrix: X + [d33, tc]
        target_cols = [c for c in ['d33', 'tc', 'sintering_temp'] if c in df.columns]
        y_matrix = df.loc[valid_indices, target_cols].values
        
        full_matrix = np.hstack((X, y_matrix))
        
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_matrix = imputer.fit_transform(full_matrix)
        
        # Extract back target predictions
        y_imputed = imputed_matrix[:, len(X[0]):]
        
        for idx_pos, df_idx in enumerate(valid_indices):
            if df_idx in target_rows: # only overwrite if user requested impute
                for col_idx, col_name in enumerate(target_cols):
                    if pd.isna(df.loc[df_idx, col_name]):
                        df.loc[df_idx, col_name] = y_imputed[idx_pos, col_idx]
                        df.loc[df_idx, f'is_{col_name}_imputed'] = True
                        
        return df
