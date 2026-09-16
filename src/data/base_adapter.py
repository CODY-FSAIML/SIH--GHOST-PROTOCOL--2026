from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
from .schema import CANONICAL_COLUMNS


class BaseDatasetAdapter(ABC):
    """
    Abstract Base Adapter for standardizing heterogeneous security datasets 
    into Normalized Network Records.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @property
    @abstractmethod
    def column_mapping(self) -> Dict[str, str]:
        """
        Maps raw dataset column names -> canonical field names.
        Example: {'Src Port': 'src_port', 'Dst IP': 'dst_ip'}
        """
        pass

    @abstractmethod
    def standardize_labels(self, raw_labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Convert dataset-specific label strings into canonical category labels and binary is_attack flags.
        Returns:
            Tuple[canonical_labels: pd.Series, is_attack: pd.Series]
        """
        pass

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a raw DataFrame into the canonical schema.
        """
        normalized_df = pd.DataFrame()

        # 1. Map columns present in column_mapping
        mapping = self.column_mapping
        for raw_col, canonical_col in mapping.items():
            if raw_col in df.columns:
                normalized_df[canonical_col] = df[raw_col]

        # 2. Ensure all canonical columns exist (fill missing with None/NaN)
        for col in CANONICAL_COLUMNS:
            if col not in normalized_df.columns:
                normalized_df[col] = None

        # 3. Standardize labels if raw label column is mapped
        raw_label_col = next((raw for raw, can in mapping.items() if can == "raw_label"), None)
        if raw_label_col and raw_label_col in df.columns:
            canonical_label, is_attack = self.standardize_labels(df[raw_label_col])
            normalized_df["label"] = canonical_label
            normalized_df["is_attack"] = is_attack
            normalized_df["raw_label"] = df[raw_label_col]

        # 4. Set dataset metadata attribute
        normalized_df["dataset_name"] = self.dataset_name

        # Reorder to standard canonical layout
        return normalized_df[CANONICAL_COLUMNS]

    def validate(self, normalized_df: pd.DataFrame) -> bool:
        """
        Verify that all canonical columns are present in the normalized DataFrame.
        """
        missing = [col for col in CANONICAL_COLUMNS if col not in normalized_df.columns]
        if missing:
            raise ValueError(f"Missing canonical columns in normalized DataFrame: {missing}")
        return True
