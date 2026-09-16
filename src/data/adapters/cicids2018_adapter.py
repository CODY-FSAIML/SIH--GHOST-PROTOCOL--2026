from typing import Dict, Tuple
import pandas as pd
from ..base_adapter import BaseDatasetAdapter


class CICIDS2018Adapter(BaseDatasetAdapter):
    """
    Adapter for CIC-IDS2018 dataset CSV files.
    """

    def __init__(self):
        super().__init__(dataset_name="CIC-IDS2018")

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            "Timestamp": "timestamp_str",
            "Src IP": "src_ip",
            "Src Port": "src_port",
            "Dst IP": "dst_ip",
            "Dst Port": "dst_port",
            "Protocol": "protocol",
            "Flow Duration": "duration",
            "TotLen Fwd Pkts": "bytes_sent",
            "TotLen Bwd Pkts": "bytes_received",
            "Tot Fwd Pkts": "packets_sent",
            "Tot Bwd Pkts": "packets_received",
            "Label": "raw_label"
        }

    def standardize_labels(self, raw_labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
        canonical_labels = []
        is_attack = []

        for val in raw_labels.astype(str):
            clean_val = val.strip().lower()
            if clean_val in ["benign"]:
                canonical_labels.append("benign")
                is_attack.append(0)
            elif "ddos" in clean_val or "dos" in clean_val:
                canonical_labels.append("dos")
                is_attack.append(1)
            elif "portscan" in clean_val or "scan" in clean_val:
                canonical_labels.append("portscan")
                is_attack.append(1)
            elif "bot" in clean_val:
                canonical_labels.append("botnet")
                is_attack.append(1)
            elif "brute" in clean_val or "force" in clean_val:
                canonical_labels.append("bruteforce")
                is_attack.append(1)
            elif "infilteration" in clean_val or "infiltration" in clean_val:
                canonical_labels.append("infiltration")
                is_attack.append(1)
            elif "web" in clean_val or "sql" in clean_val or "xss" in clean_val:
                canonical_labels.append("web_attack")
                is_attack.append(1)
            else:
                canonical_labels.append("unknown_attack")
                is_attack.append(1)

        return pd.Series(canonical_labels, index=raw_labels.index), pd.Series(is_attack, index=raw_labels.index)
