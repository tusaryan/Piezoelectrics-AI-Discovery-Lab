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
        train, test = train_test_split(frame, test_size=test_size, random_state=42, shuffle=True)
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def apply_test_set_policy(self, test_frame: pd.DataFrame, required_cols: list[str]) -> tuple[pd.DataFrame, str]:
        before = len(test_frame)
        numeric_cols = [col for col in required_cols if col in test_frame.columns and col != "formula"]
        pruned = test_frame.copy()
        for col in numeric_cols:
            pruned[col] = pd.to_numeric(pruned[col], errors="coerce")
        pruned = pruned.dropna(subset=required_cols).reset_index(drop=True)
        dropped = before - len(pruned)
        log_line = f"Test set: dropped {dropped}/{before} rows with missing/invalid values"
        return pruned, log_line
