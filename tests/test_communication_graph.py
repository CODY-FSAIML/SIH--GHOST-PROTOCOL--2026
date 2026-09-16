# tests/test_communication_graph.py
"""Tests for src.graph.communication_graph.build_communication_graph
using the canonical normalized flow schema.
"""

import pandas as pd
import numpy as np
import unittest

# Ensure project root is on path
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.graph.communication_graph import (
    build_communication_graph,
    build_communication_graph_reference,
)

class TestCommunicationGraphCanonical(unittest.TestCase):
    def setUp(self):
        # Common canonical columns for all tests
        self.base_columns = [
            "src_ip",
            "dst_ip",
            "bytes_sent",
            "bytes_received",
            "packets_sent",
            "packets_received",
            "protocol",
            "duration",
        ]

    def _make_df(self, rows):
        """Utility to create DataFrame from list of dicts, ensuring all columns present."""
        df = pd.DataFrame(rows)
        # Ensure all base columns exist (may be missing in specific tests)
        for col in self.base_columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype=float)
        return df

    def test_deterministic_node_ordering(self):
        rows = [
            {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.1", "bytes_sent": 100, "bytes_received": 200,
             "packets_sent": 1, "packets_received": 2, "protocol": "TCP", "duration": 0.5},
            {"src_ip": "10.0.0.3", "dst_ip": "10.0.0.2", "bytes_sent": 50, "bytes_received": 70,
             "packets_sent": 1, "packets_received": 1, "protocol": "UDP", "duration": 0.2},
        ]
        df = self._make_df(rows)
        result = build_communication_graph(df)
        # Nodes should be sorted alphabetically
        expected_node_ids = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        self.assertEqual(result["node_ids"], expected_node_ids)

    def test_directed_edge_construction_and_aggregation(self):
        rows = [
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "bytes_sent": 300, "bytes_received": 150,
             "packets_sent": 3, "packets_received": 1, "protocol": "TCP", "duration": 1.0},
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "bytes_sent": 200, "bytes_received": 100,
             "packets_sent": 2, "packets_received": 1, "protocol": "TCP", "duration": 0.5},
            {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.1", "bytes_sent": 400, "bytes_received": 250,
             "packets_sent": 4, "packets_received": 2, "protocol": "UDP", "duration": 0.8},
        ]
        df = self._make_df(rows)
        result = build_communication_graph(df)
        # Edge meta should contain two directed edges
        self.assertEqual(len(result["edge_meta"]), 2)
        # Determine node index mapping
        src_ip_to_idx = {ip: idx for idx, ip in enumerate(result["node_ids"])}
        edge_index = result["edge_index"]
        # Edge (1->2) and (2->1) must both be present (order may vary)
        idx_1 = src_ip_to_idx["10.0.0.1"]
        idx_2 = src_ip_to_idx["10.0.0.2"]
        # Verify that both directed pairs appear in edge_index
        edge_pairs = set((edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1]))
        self.assertIn((idx_1, idx_2), edge_pairs)
        self.assertIn((idx_2, idx_1), edge_pairs)
        # Check aggregation values for edge 1->2 (first entry in edge_meta corresponding to that direction)
        meta = result["edge_meta"]
        idx_edge = meta.index(("10.0.0.1", "10.0.0.2"))
        edge_feat = result["edge_features"]
        flow_count = edge_feat[idx_edge, 0]
        total_bytes = edge_feat[idx_edge, 1]
        total_packets = edge_feat[idx_edge, 2]
        mean_dur = edge_feat[idx_edge, 3]
        tcp_ratio = edge_feat[idx_edge, 4]
        udp_ratio = edge_feat[idx_edge, 5]
        self.assertEqual(flow_count, 2)
        self.assertAlmostEqual(total_bytes, 300 + 200)
        self.assertAlmostEqual(total_packets, 3 + 2)
        self.assertAlmostEqual(mean_dur, (1.0 + 0.5) / 2)
        self.assertAlmostEqual(tcp_ratio, 1.0)
        self.assertAlmostEqual(udp_ratio, 0.0)

    def test_node_feature_aggregation(self):
        rows = [
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "bytes_sent": 100, "bytes_received": 90,
             "packets_sent": 1, "packets_received": 2, "protocol": "TCP", "duration": 0.1},
            {"src_ip": "10.0.0.3", "dst_ip": "10.0.0.1", "bytes_sent": 50, "bytes_received": 70,
             "packets_sent": 2, "packets_received": 1, "protocol": "UDP", "duration": 0.2},
        ]
        df = self._make_df(rows)
        result = build_communication_graph(df)
        node_ids = result["node_ids"]
        features = result["node_features"]
        idx = {ip: i for i, ip in enumerate(node_ids)}
        f1 = features[idx["10.0.0.1"]]
        self.assertEqual(f1[0], 1)  # in_degree (from 10.0.0.3)
        self.assertEqual(f1[1], 1)  # out_degree (to 10.0.0.2)
        self.assertAlmostEqual(f1[2], 70)  # in_bytes (bytes_received)
        self.assertAlmostEqual(f1[3], 100)  # out_bytes (bytes_sent)
        self.assertAlmostEqual(f1[4], 1)  # in_packets (packets_received)
        self.assertAlmostEqual(f1[5], 1)  # out_packets (packets_sent)

    def test_empty_window(self):
        df = pd.DataFrame(columns=self.base_columns)
        result = build_communication_graph(df)
        self.assertEqual(result["node_ids"], [])
        self.assertEqual(result["node_features"].shape, (0, 6))
        self.assertEqual(result["edge_index"].shape, (2, 0))
        self.assertEqual(result["edge_features"].shape, (0, 6))

    def test_label_leakage(self):
        rows = [
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "bytes_sent": 100, "bytes_received": 200,
             "packets_sent": 1, "packets_received": 2, "protocol": "TCP", "duration": 0.5,
             "label": "botnet", "raw_label": "flow=Botnet", "is_attack": 1},
        ]
        df_no_label = self._make_df([{k: v for k, v in rows[0].items() if k not in {"label", "raw_label", "is_attack"}}])
        df_with_label = self._make_df(rows)
        res1 = build_communication_graph(df_no_label)
        res2 = build_communication_graph(df_with_label)
        self.assertEqual(res1["node_ids"], res2["node_ids"])
        np.testing.assert_allclose(res1["node_features"], res2["node_features"])
        np.testing.assert_allclose(res1["edge_index"], res2["edge_index"])
        np.testing.assert_allclose(res1["edge_features"], res2["edge_features"])
        self.assertEqual(res1["edge_meta"], res2["edge_meta"])

    def test_optimized_graph_matches_reference_on_five_windows(self):
        """The optimized path preserves every graph array on five windows.

        Numeric arrays use rtol=1e-12 and atol=1e-12; IDs and edge indices are exact.
        """
        windows = [
            self._make_df([
                {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "bytes_sent": 10, "bytes_received": 20,
                 "packets_sent": 1, "packets_received": 2, "protocol": "TCP", "duration": 0.5},
            ]),
            self._make_df([
                {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "bytes_sent": 30, "bytes_received": 40,
                 "packets_sent": 3, "packets_received": 4, "protocol": "UDP", "duration": 1.5},
                {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "bytes_sent": 5, "bytes_received": 6,
                 "packets_sent": 1, "packets_received": 1, "protocol": "TCP", "duration": 0.5},
            ]),
            self._make_df([
                {"src_ip": "10.0.0.4", "dst_ip": "10.0.0.1", "bytes_sent": 0, "bytes_received": 0,
                 "packets_sent": 0, "packets_received": 0, "protocol": "icmp", "duration": None},
            ]),
            pd.DataFrame(columns=self.base_columns),
            self._make_df([
                {"src_ip": "10.0.0.5", "dst_ip": "10.0.0.6", "bytes_sent": "bad", "bytes_received": 8,
                 "packets_sent": 2, "packets_received": "bad", "protocol": " TCP ", "duration": "bad",
                 "total_bytes": 9, "total_packets": 3},
            ]),
        ]
        for df_window in windows:
            reference = build_communication_graph_reference(df_window)
            optimized = build_communication_graph(df_window)
            self.assertEqual(reference["node_ids"], optimized["node_ids"])
            np.testing.assert_allclose(reference["node_features"], optimized["node_features"], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(reference["edge_index"], optimized["edge_index"])
            np.testing.assert_allclose(reference["edge_features"], optimized["edge_features"], rtol=1e-12, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
