from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass
class NormalizedNetworkRecord:
    """
    Unified Schema for Network Traffic and Security Events across heterogeneous datasets.
    Fields are intentionally Optional to support diverse data modalities (Flow, Packet, Auth logs).
    """
    timestamp: Optional[float] = None          # POSIX timestamp (seconds since epoch)
    timestamp_str: Optional[str] = None        # ISO formatted timestamp string

    src_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_ip: Optional[str] = None
    dst_port: Optional[int] = None
    protocol: Optional[str] = None             # TCP, UDP, ICMP, etc.

    duration: Optional[float] = None           # Flow/connection duration in seconds
    bytes_sent: Optional[int] = None
    bytes_received: Optional[int] = None
    packets_sent: Optional[int] = None
    packets_received: Optional[int] = None

    flow_id: Optional[str] = None
    dataset_name: Optional[str] = None          # e.g., 'CIC-IDS2018', 'UNSW-NB15'

    label: Optional[str] = None                # Standardized label ('benign', 'dos', 'portscan', etc.)
    raw_label: Optional[str] = None            # Original label string from dataset
    is_attack: Optional[int] = None            # 0 for benign, 1 for attack, None if unknown

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizedNetworkRecord":
        """Instantiate record from dictionary matching dataclass fields."""
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


CANONICAL_COLUMNS: List[str] = [
    "timestamp",
    "timestamp_str",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "duration",
    "bytes_sent",
    "bytes_received",
    "packets_sent",
    "packets_received",
    "flow_id",
    "dataset_name",
    "label",
    "raw_label",
    "is_attack"
]
