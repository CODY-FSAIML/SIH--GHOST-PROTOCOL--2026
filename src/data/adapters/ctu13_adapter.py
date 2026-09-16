"""
CTU-13 Dataset Adapter
======================
Maps raw CTU-13 Scenario binetflow CSV fields into the canonical
NormalizedNetworkRecord schema defined in src/data/schema.py.

Design contract:  ctu13_normalization_contract.md
Inspected file:   data/raw/ctu13/capture20110810.binetflow.txt
Actual columns:   StartTime, Dur, Proto, SrcAddr, Sport, Dir,
                  DstAddr, Dport, State, sTos, dTos,
                  TotPkts, TotBytes, SrcBytes, Label

IMPORTANT SEMANTIC DECISIONS:
- TotPkts / TotBytes are bidirectional totals → stored in metadata only.
  They are NOT mapped to packets_sent / bytes_sent to avoid semantic corruption.
- DstBytes is safely derived as TotBytes - SrcBytes (validated).
- Sport / Dport may be hex strings (ICMP) → converted to int; raw preserved.
- Dir has leading whitespace → stripped and mapped to clean enum.
- State is an Argus compound lifecycle string → preserved raw in metadata.
- Labels are 113 structured strings → classified by token into 3 categories.
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from ..base_adapter import BaseDatasetAdapter
from ..schema import CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# Label classification helpers
# ---------------------------------------------------------------------------

def _classify_ctu13_label(raw: str) -> Tuple[str, Optional[int]]:
    """
    Classify a single raw CTU-13 label string into a canonical category
    and binary is_attack flag.

    CTU-13 label structure:
        flow=<Direction>-<Category>-<Variant>-<Protocol/Action>

    Classification is based on recognising the exact token 'Botnet',
    'Normal', or 'Background' in the token list after stripping the
    'flow=' prefix and splitting on '-'.

    Returns:
        (canonical_label, is_attack)
        canonical_label: 'botnet' | 'normal' | 'background' | 'unknown'
        is_attack: 1 for botnet, 0 for normal/background, None for unknown
    """
    if not isinstance(raw, str):
        return "unknown", None

    # Strip 'flow=' prefix (always present in CTU-13)
    stripped = raw.strip()
    if stripped.startswith("flow="):
        stripped = stripped[5:]

    # Split on '-' to get individual tokens
    tokens = stripped.split("-")

    if "Botnet" in tokens:
        return "botnet", 1
    elif "Normal" in tokens:
        return "normal", 0
    elif "Background" in tokens:
        return "background", 0
    else:
        return "unknown", None


def _extract_label_metadata(raw: str) -> Dict:
    """
    Extract botnet variant (e.g. 'V42') and protocol_action
    (e.g. 'UDP-DNS', 'TCP-Attempt-SPAM') from the raw CTU-13 label.

    These are preserved in metadata for eventual MITRE ATT&CK mapping.
    """
    result = {}
    if not isinstance(raw, str):
        return result

    stripped = raw.strip()
    if stripped.startswith("flow="):
        stripped = stripped[5:]

    tokens = stripped.split("-")

    # botnet_variant: token starting with 'V' followed by digits (e.g. V42)
    for token in tokens:
        if len(token) >= 2 and token[0] == "V" and token[1:].isdigit():
            result["botnet_variant"] = token
            break

    # protocol_action: everything after the variant token
    # e.g. 'From-Botnet-V42-UDP-DNS' → 'UDP-DNS'
    #      'From-Normal-V42-Stribrek' → 'Stribrek'
    if "botnet_variant" in result:
        variant_idx = next(
            (i for i, t in enumerate(tokens) if t == result["botnet_variant"]),
            None,
        )
        if variant_idx is not None and variant_idx + 1 < len(tokens):
            result["protocol_action"] = "-".join(tokens[variant_idx + 1 :])

    return result


# ---------------------------------------------------------------------------
# Port normalization helper
# ---------------------------------------------------------------------------

def _normalize_port(value) -> Tuple[Optional[int], Optional[str]]:
    """
    Safely convert a CTU-13 Sport/Dport value to an integer.

    Rules:
    - NaN / None / empty string → (None, None)
    - Hex strings '0x...' / '0X...' → (int(value, 16), raw_string)
    - Decimal numeric strings → (int(value), None)
    - Anything else → (None, str(value)) preserving the raw

    Returns:
        (port_as_int_or_None, raw_string_or_None)
    """
    # Missing
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None

    raw_str = str(value).strip()

    if raw_str == "" or raw_str.lower() == "nan":
        return None, None

    # Hexadecimal
    if raw_str.lower().startswith("0x"):
        try:
            return int(raw_str, 16), raw_str
        except ValueError:
            return None, raw_str

    # Decimal
    if raw_str.isdigit():
        return int(raw_str), None

    # Unexpected (e.g. partial values from corrupt rows)
    return None, raw_str


# ---------------------------------------------------------------------------
# Direction normalization helper
# ---------------------------------------------------------------------------

_DIR_MAP = {
    "<->": "bidirectional",
    "->": "outbound",
    "<-": "inbound",
    "<?>": "unknown",
}


def _normalize_direction(raw_dir) -> Tuple[str, bool]:
    """
    Strip whitespace from Dir and map to a clean canonical string.

    Returns:
        (flow_direction, is_bidirectional)
    """
    if not isinstance(raw_dir, str):
        return "unknown", False

    cleaned = raw_dir.strip()
    direction = _DIR_MAP.get(cleaned, "unknown")
    is_bidirectional = direction == "bidirectional"
    return direction, is_bidirectional


# ---------------------------------------------------------------------------
# State category helper
# ---------------------------------------------------------------------------

def _state_category(state: str) -> str:
    """
    Map an Argus compound state string to a coarse lifecycle category.

    These categories are deliberately conservative — if the state is not
    clearly mappable, 'other' is returned rather than guessing.
    """
    if not isinstance(state, str):
        return "other"

    upper = state.strip().upper()

    # Ongoing / established
    if upper in ("CON", "PA_PA", "PA_A", "A_PA", "A_A"):
        return "established"
    if upper.startswith("FPA") or upper.startswith("PA"):
        return "established"

    # Attempt / incomplete
    if upper in ("SYN", "REQ", "INT"):
        return "attempt"

    # Graceful close
    if "FA" in upper or "FIN" in upper:
        return "closed"

    # Reset
    if "RST" in upper:
        return "reset"

    return "other"


# ---------------------------------------------------------------------------
# CTU-13 Adapter
# ---------------------------------------------------------------------------

class CTU13Adapter(BaseDatasetAdapter):
    """
    Adapter for CTU-13 Scenario binetflow CSV files.

    Inherits from BaseDatasetAdapter and overrides normalize() to perform
    CTU-13-specific transformations that cannot be handled by the base
    class's generic column-rename logic:

        - Port hex→int conversion (ICMP uses hex port encoding)
        - Bidirectional total bytes/packets preserved in metadata only
        - DstBytes derived as TotBytes - SrcBytes
        - Dir whitespace stripped and mapped to clean direction enum
        - State preserved as raw Argus string + coarse category
        - 113 raw labels classified to 3 top-level categories via token logic

    The adapter does NOT load data itself.
    Call normalize(df) with a pandas DataFrame of raw rows.
    """

    def __init__(self):
        super().__init__(dataset_name="CTU-13")

    # ------------------------------------------------------------------
    # Required abstract properties / methods from BaseDatasetAdapter
    # ------------------------------------------------------------------

    @property
    def column_mapping(self) -> Dict[str, str]:
        """
        Minimal direct column renames used by the base class.
        Complex transformations (ports, bytes, direction, state, metadata)
        are handled in the overridden normalize() below.
        """
        return {
            "StartTime": "timestamp_str",
            "Dur": "duration",
            "Proto": "protocol",
            "SrcAddr": "src_ip",
            "DstAddr": "dst_ip",
            "Label": "raw_label",
        }

    def standardize_labels(
        self, raw_labels: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Classify CTU-13 raw label strings into canonical categories.

        canonical_label: 'botnet' | 'normal' | 'background' | 'unknown'
        is_attack:       1 (botnet) | 0 (normal/background) | None (unknown)
        """
        canonical_list = []
        is_attack_list = []

        for raw in raw_labels:
            cat, is_atk = _classify_ctu13_label(raw)
            canonical_list.append(cat)
            is_attack_list.append(is_atk)

        return (
            pd.Series(canonical_list, index=raw_labels.index, dtype=object),
            pd.Series(is_attack_list, index=raw_labels.index, dtype=object),
        )

    # ------------------------------------------------------------------
    # Overridden normalize() — CTU-13-specific row-level processing
    # ------------------------------------------------------------------

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a raw CTU-13 binetflow DataFrame into the canonical schema.

        Overrides BaseDatasetAdapter.normalize() because CTU-13 requires:
          1. Row-level port hex→int conversion
          2. Row-level byte derivation (dst_bytes = total_bytes - src_bytes)
          3. Row-level direction parsing
          4. Per-row metadata dict construction
          5. Correct semantic placement of bidirectional totals into metadata

        After transformation, canonical columns are reordered as required
        by CANONICAL_COLUMNS and validate() is called automatically.
        """
        n = len(df)
        out = pd.DataFrame(index=df.index)

        # ---- Timestamps ------------------------------------------------
        out["timestamp_str"] = df["StartTime"].astype(str)
        ts_parsed = pd.to_datetime(
            df["StartTime"],
            format="%Y/%m/%d %H:%M:%S.%f",
            errors="coerce",
        )
        # POSIX float (seconds since epoch), timezone-naive
        out["timestamp"] = ts_parsed.astype(np.int64) / 1e9

        # ---- Simple renames --------------------------------------------
        out["duration"] = pd.to_numeric(df["Dur"], errors="coerce")
        out["protocol"] = df["Proto"].astype(str).str.lower().str.strip()
        out["src_ip"] = df["SrcAddr"].astype(str).str.strip()
        out["dst_ip"] = df["DstAddr"].astype(str).str.strip()

        # ---- Ports (hex-safe) -----------------------------------------
        src_ports, src_ports_raw = zip(
            *[_normalize_port(v) for v in df["Sport"]]
        )
        dst_ports, dst_ports_raw = zip(
            *[_normalize_port(v) for v in df["Dport"]]
        )
        out["src_port"] = list(src_ports)
        out["dst_port"] = list(dst_ports)

        # ---- Canonical schema fields left intentionally None -----------
        # bytes_sent, bytes_received: CTU-13 gives only TotBytes (bidirectional)
        #   → mapping to bytes_sent would be semantically incorrect
        # packets_sent, packets_received: CTU-13 gives only TotPkts (bidirectional)
        #   → no per-direction packet breakdown available
        out["bytes_sent"] = None
        out["bytes_received"] = None
        out["packets_sent"] = None
        out["packets_received"] = None

        # ---- Remaining schema scalars ---------------------------------
        out["flow_id"] = None
        out["dataset_name"] = self.dataset_name

        # ---- Labels ---------------------------------------------------
        raw_labels = df["Label"].astype(str)
        out["raw_label"] = raw_labels

        canonical_labels, is_attack_flags = self.standardize_labels(raw_labels)
        out["label"] = canonical_labels
        out["is_attack"] = is_attack_flags

        # ---- Build per-row metadata dicts ------------------------------
        # Computed ahead of the loop to avoid pandas row-by-row overhead
        tot_pkts = pd.to_numeric(df["TotPkts"], errors="coerce")
        tot_bytes = pd.to_numeric(df["TotBytes"], errors="coerce")
        src_bytes_col = pd.to_numeric(df["SrcBytes"], errors="coerce")

        # dst_bytes derived: safe only when tot_bytes >= src_bytes
        dst_bytes_col = np.where(
            (tot_bytes.notna()) & (src_bytes_col.notna()) & (tot_bytes >= src_bytes_col),
            tot_bytes - src_bytes_col,
            np.nan,
        )

        sTos_col = pd.to_numeric(df["sTos"], errors="coerce")
        dTos_col = pd.to_numeric(df["dTos"], errors="coerce")
        state_col = df["State"].astype(str).str.strip()

        directions_cleaned = []
        is_bidir_list = []
        for raw_dir in df["Dir"]:
            d, b = _normalize_direction(raw_dir)
            directions_cleaned.append(d)
            is_bidir_list.append(b)

        metadata_list = []
        for i in range(n):
            raw_label_val = raw_labels.iloc[i]
            state_val = state_col.iloc[i]

            # Safely convert numpy scalars to Python native types
            def _safe(v):
                if v is None:
                    return None
                try:
                    if np.isnan(float(v)):
                        return None
                    return float(v) if "." in str(v) else int(float(v))
                except (TypeError, ValueError):
                    return None

            meta = {
                # Bidirectional flow totals (semantically correct placement)
                "total_packets": _safe(tot_pkts.iloc[i]),
                "total_bytes": _safe(tot_bytes.iloc[i]),
                "src_bytes": _safe(src_bytes_col.iloc[i]),
                "dst_bytes": _safe(dst_bytes_col[i]),
                # Connection state
                "connection_state": state_val if state_val != "nan" else None,
                "state_category": _state_category(state_val),
                # Direction
                "flow_direction": directions_cleaned[i],
                "is_bidirectional": is_bidir_list[i],
                # Type of Service
                "sTos": _safe(sTos_col.iloc[i]),
                "dTos": _safe(dTos_col.iloc[i]),
                # Raw port strings when conversion was needed
                "sport_raw": src_ports_raw[i],
                "dport_raw": dst_ports_raw[i],
            }

            # Label-derived metadata
            meta.update(_extract_label_metadata(raw_label_val))

            metadata_list.append(meta)

        out["metadata"] = metadata_list

        # ---- Reorder to CANONICAL_COLUMNS layout ----------------------
        # metadata is NOT in CANONICAL_COLUMNS (it's a dict field on the
        # dataclass), so we carry it separately and re-attach after reorder.
        canonical_out = pd.DataFrame(index=df.index)
        for col in CANONICAL_COLUMNS:
            canonical_out[col] = out[col] if col in out.columns else None

        # Re-attach metadata column (not in CANONICAL_COLUMNS but required)
        canonical_out["metadata"] = metadata_list

        return canonical_out

    def validate(self, normalized_df: pd.DataFrame) -> bool:
        """Verify all canonical columns are present."""
        missing = [
            col for col in CANONICAL_COLUMNS if col not in normalized_df.columns
        ]
        if missing:
            raise ValueError(
                f"[CTU13Adapter] Missing canonical columns: {missing}"
            )
        return True
