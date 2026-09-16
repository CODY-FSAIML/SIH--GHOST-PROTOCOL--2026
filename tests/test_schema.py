from src.data.schema import NormalizedNetworkRecord, CANONICAL_COLUMNS
from src.data.adapters.cicids2018_adapter import CICIDS2018Adapter
from src.features.window_builder import TimeWindowBuilder
import pandas as pd


def test_schema_instantiation():
    record = NormalizedNetworkRecord(
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        dst_port=80,
        protocol="TCP",
        is_attack=0,
        label="benign"
    )
    d = record.to_dict()
    assert d["src_ip"] == "192.168.1.10"
    assert d["dst_port"] == 80
    assert d["src_port"] is None  # Optional field defaults to None


def test_cicids2018_adapter():
    raw_df = pd.DataFrame([
        {
            "Timestamp": "2018-02-14 08:31:00",
            "Src IP": "192.168.1.5",
            "Src Port": 443,
            "Dst IP": "10.0.0.2",
            "Dst Port": 80,
            "Protocol": 6,
            "Flow Duration": 1.5,
            "TotLen Fwd Pkts": 500,
            "TotLen Bwd Pkts": 1200,
            "Tot Fwd Pkts": 5,
            "Tot Bwd Pkts": 10,
            "Label": "Benign"
        },
        {
            "Timestamp": "2018-02-14 08:31:05",
            "Src IP": "172.16.0.1",
            "Src Port": 52144,
            "Dst IP": "10.0.0.2",
            "Dst Port": 80,
            "Protocol": 6,
            "Flow Duration": 0.2,
            "TotLen Fwd Pkts": 2000,
            "TotLen Bwd Pkts": 0,
            "Tot Fwd Pkts": 20,
            "Tot Bwd Pkts": 0,
            "Label": "DDoS attacks-LOIC-HTTP"
        }
    ])

    adapter = CICIDS2018Adapter()
    norm_df = adapter.normalize(raw_df)

    assert adapter.validate(norm_df)
    assert len(norm_df) == 2
    assert norm_df.iloc[0]["label"] == "benign"
    assert norm_df.iloc[0]["is_attack"] == 0
    assert norm_df.iloc[1]["label"] == "dos"
    assert norm_df.iloc[1]["is_attack"] == 1
    assert norm_df.iloc[1]["dataset_name"] == "CIC-IDS2018"


def test_time_window_builder():
    raw_df = pd.DataFrame([
        {
            "Timestamp": "2018-02-14 08:31:00",
            "Src IP": "192.168.1.5",
            "Dst IP": "10.0.0.2",
            "Dst Port": 80,
            "TotLen Fwd Pkts": 500,
            "TotLen Bwd Pkts": 500,
            "Tot Fwd Pkts": 5,
            "Tot Bwd Pkts": 5,
            "Label": "Benign"
        },
        {
            "Timestamp": "2018-02-14 08:31:02",
            "Src IP": "172.16.0.1",
            "Dst IP": "10.0.0.2",
            "Dst Port": 80,
            "TotLen Fwd Pkts": 1000,
            "TotLen Bwd Pkts": 0,
            "Tot Fwd Pkts": 10,
            "Tot Bwd Pkts": 0,
            "Label": "DDoS"
        }
    ])

    adapter = CICIDS2018Adapter()
    norm_df = adapter.normalize(raw_df)

    builder = TimeWindowBuilder(window_size_sec=10.0)
    windows = builder.build_windows(norm_df)

    assert len(windows) >= 1
    w0 = windows[0]
    assert w0["record_count"] == 2
    assert w0["is_attack_window"] == 1
    assert w0["total_bytes"] == 2000
