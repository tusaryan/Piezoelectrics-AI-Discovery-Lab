from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from piezo_ml.pipeline.data_cleaner import clean_dataframe


class DataLoaderValidator:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "formula" not in frame.columns:
            raise ValueError("Dataset must include formula column")
        if "uid" not in frame.columns:
            frame = frame.copy()
            frame.insert(0, "uid", range(1, len(frame) + 1))
        return frame

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned, _ = clean_dataframe(frame)
        return cleaned

    def split_train_test(self, frame: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        import re
        import numpy as np

        def get_elements(formula):
            # Extract basic element symbols to create a chemical system group
            return "-".join(sorted(set(re.findall(r'[A-Z][a-z]?', str(formula)))))

        df = frame.copy()
        if "formula" in df.columns:
            df['__chem_group'] = df['formula'].apply(get_elements)
        else:
            df['__chem_group'] = 'all'

        # Calculate counts to find singletons
        group_counts = df['__chem_group'].value_counts()
        
        # Groups with size 1 cannot be stratified. Put them in train to ensure they are learned.
        # Groups with size small might throw errors if test_size * count < 1. 
        # train_test_split in sklearn allows stratify if minimum class > 1.
        singletons = group_counts[group_counts < 2].index
        train_singletons = df[df['__chem_group'].isin(singletons)]
        multiples = df[~df['__chem_group'].isin(singletons)]

        if len(multiples) > 0:
            n_classes = multiples['__chem_group'].nunique()
            n_test_samples = max(1, int(len(multiples) * test_size))
            
            stratify_arg = multiples['__chem_group'] if n_test_samples >= n_classes else None
            
            train_mult, test_mult = train_test_split(
                multiples,
                test_size=test_size,
                random_state=42,
                shuffle=True,
                stratify=stratify_arg
            )
        else:
            train_mult = pd.DataFrame(columns=df.columns)
            test_mult = pd.DataFrame(columns=df.columns)

        train = pd.concat([train_singletons, train_mult]).drop(columns=['__chem_group'])
        test = test_mult.drop(columns=['__chem_group']) if not test_mult.empty else pd.DataFrame(columns=train.columns)

        # Shuffle train again because singletons were appended at the top
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        return train, test.reset_index(drop=True)

    def apply_test_set_policy(self, test_frame: pd.DataFrame, required_cols: list[str]) -> tuple[pd.DataFrame, str]:
        before = len(test_frame)
        numeric_cols = [col for col in required_cols if col in test_frame.columns and col != "formula"]
        pruned = test_frame.copy()
        
        for col in numeric_cols:
            pruned[col] = pd.to_numeric(pruned[col], errors="coerce")
            
        # Drop if formula is missing
        if "formula" in required_cols and "formula" in pruned.columns:
            pruned = pruned.dropna(subset=["formula"])
            
        # Drop only if ALL numeric target values are missing
        if numeric_cols:
            pruned = pruned.dropna(subset=numeric_cols, how="all")
            
        pruned = pruned.reset_index(drop=True)
        dropped = before - len(pruned)
        log_line = f"Test set: dropped {dropped}/{before} rows with completely missing targets"
        return pruned, log_line
