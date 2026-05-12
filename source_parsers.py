# NOTE:
# SplunkParser, Rapid7Parser, and RadarParser are boilerplate starter parsers.
# They provide common identifier extraction patterns, but real deployments should
# adjust the regexes, metadata mappings, and anomaly logic to match the actual
# log schema and exported fields of each source.

import os
import re
import pandas as pd
from typing import Dict, List, Optional, Set


class BaseLogParser:
    """
    Base parser for different log sources.

    New log types should subclass this and override:
    - extract_ids_from_line()
    - load_artifacts()
    - enrich_chunk()
    """

    name = "base"

    def load_artifacts(self, metadata_dir: Optional[str] = None, **kwargs) -> Dict:
        return {}

    def extract_ids_from_line(self, line: str) -> Set[str]:
        return set()

    def enrich_chunk(
        self,
        chunk_text: str,
        identifiers: Set[str],
        artifacts: Dict,
    ) -> Dict:
        return {
            "source_type": self.name,
            "identifiers": sorted(list(identifiers)),
            "source_metadata": {},
        }


class HDFSParser(BaseLogParser):
    name = "hdfs"

    BLOCK_ID_REGEX = re.compile(r"blk_[\-\d]+")

    def load_artifacts(
        self,
        metadata_dir: Optional[str] = None,
        anomaly_labels: Optional[str] = None,
        event_traces: Optional[str] = None,
        event_occurrence: Optional[str] = None,
        log_templates: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        if metadata_dir:
            anomaly_labels = anomaly_labels or os.path.join(metadata_dir, "anomaly_label.csv")
            event_traces = event_traces or os.path.join(metadata_dir, "Event_traces.csv")
            event_occurrence = event_occurrence or os.path.join(metadata_dir, "Event_occurrence_matrix.csv")
            log_templates = log_templates or os.path.join(metadata_dir, "HDFS.log_templates.csv")

        return {
            "labels": self._load_anomaly_labels(anomaly_labels),
            "event_traces": self._load_event_traces(event_traces),
            "event_occurrence": self._load_event_occurrence_matrix(event_occurrence),
            "log_templates": self._load_log_templates(log_templates),
        }

    def extract_ids_from_line(self, line: str) -> Set[str]:
        return set(self.BLOCK_ID_REGEX.findall(line))

    def enrich_chunk(
        self,
        chunk_text: str,
        identifiers: Set[str],
        artifacts: Dict,
    ) -> Dict:
        block_ids_sorted = sorted(list(identifiers))

        labels = artifacts.get("labels", {})
        event_traces = artifacts.get("event_traces", {})
        event_occurrence = artifacts.get("event_occurrence", {})
        log_templates = artifacts.get("log_templates", {})

        has_anomaly = any(labels.get(bid, "Normal") == "Anomaly" for bid in block_ids_sorted)

        trace_data = {}
        occurrence_data = {}
        event_templates = {}
        event_ids = set()

        for bid in block_ids_sorted:
            if bid in event_traces:
                trace_info = event_traces[bid]
                trace_data[bid] = trace_info
                event_ids.update(self._extract_event_ids(trace_info.get("features", "")))

            if bid in event_occurrence:
                occ_info = event_occurrence[bid]
                counts = occ_info.get("event_counts", {})
                occurrence_data[bid] = {
                    "occurrence_label": occ_info.get("occurrence_label", ""),
                    "occurrence_type": occ_info.get("occurrence_type", ""),
                    "top_event_counts": self._top_event_counts(counts, top_n=5),
                }
                event_ids.update(counts.keys())

        for event_id in sorted(event_ids):
            if event_id in log_templates:
                event_templates[event_id] = log_templates[event_id]

        return {
            "source_type": self.name,
            "identifiers": block_ids_sorted,
            "block_ids": block_ids_sorted,
            "has_anomaly": has_anomaly,
            "trace_data": trace_data,
            "occurrence_data": occurrence_data,
            "event_templates": event_templates,
            "source_metadata": {
                "primary_identifier_type": "BlockId",
                "num_identifiers": len(block_ids_sorted),
            },
        }

    def _load_anomaly_labels(self, path: Optional[str]) -> Dict[str, str]:
        if path is None or not os.path.exists(path):
            print("[parser:hdfs] No anomaly_label.csv found.")
            return {}
        df = pd.read_csv(path)
        return dict(zip(df["BlockId"].astype(str), df["Label"].astype(str)))

    def _load_event_traces(self, path: Optional[str]) -> Dict[str, Dict]:
        if path is None or not os.path.exists(path):
            print("[parser:hdfs] No Event_traces.csv found.")
            return {}

        df = pd.read_csv(path)
        traces = {}

        for _, row in df.iterrows():
            traces[str(row["BlockId"])] = {
                "trace_label": str(row["Label"]) if pd.notna(row["Label"]) else "",
                "trace_type": str(row["Type"]) if pd.notna(row["Type"]) else "",
                "features": str(row["Features"]) if pd.notna(row["Features"]) else "",
                "time_interval": str(row["TimeInterval"]) if pd.notna(row["TimeInterval"]) else "",
                "latency": str(row["Latency"]) if pd.notna(row["Latency"]) else "",
            }

        return traces

    def _load_event_occurrence_matrix(self, path: Optional[str]) -> Dict[str, Dict]:
        if path is None or not os.path.exists(path):
            print("[parser:hdfs] No Event_occurrence_matrix.csv found.")
            return {}

        df = pd.read_csv(path)
        occurrence = {}
        event_cols = [c for c in df.columns if c.startswith("E")]

        for _, row in df.iterrows():
            block_id = str(row["BlockId"])
            counts = {}

            for col in event_cols:
                val = row[col]
                if pd.notna(val):
                    try:
                        val_int = int(val)
                        if val_int > 0:
                            counts[col] = val_int
                    except Exception:
                        pass

            occurrence[block_id] = {
                "occurrence_label": str(row["Label"]) if pd.notna(row["Label"]) else "",
                "occurrence_type": str(row["Type"]) if pd.notna(row["Type"]) else "",
                "event_counts": counts,
            }

        return occurrence

    def _load_log_templates(self, path: Optional[str]) -> Dict[str, str]:
        if path is None or not os.path.exists(path):
            print("[parser:hdfs] No HDFS.log_templates.csv found.")
            return {}

        df = pd.read_csv(path)
        return dict(zip(df["EventId"].astype(str), df["EventTemplate"].astype(str)))

    def _extract_event_ids(self, features_str: str) -> List[str]:
        if not features_str:
            return []
        return re.findall(r"E\d+", features_str)

    def _top_event_counts(self, event_counts: Dict[str, int], top_n: int = 5) -> Dict[str, int]:
        return dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])


class SplunkParser(BaseLogParser):
    """
    Boilerplate parser for Splunk-exported logs.

    Adjust regexes and metadata fields based on your actual Splunk export format.
    Common fields may include host, sourcetype, source, index, severity, user, src_ip, dest_ip.
    """

    name = "splunk"

    IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    USER_REGEX = re.compile(r"(?:user|username|account)=([A-Za-z0-9_.@\-]+)", re.IGNORECASE)
    HOST_REGEX = re.compile(r"(?:host|hostname)=([A-Za-z0-9_.\-]+)", re.IGNORECASE)

    def extract_ids_from_line(self, line: str) -> Set[str]:
        ids = set()
        ids.update(self.IP_REGEX.findall(line))
        ids.update(self.USER_REGEX.findall(line))
        ids.update(self.HOST_REGEX.findall(line))
        return ids

    def enrich_chunk(self, chunk_text: str, identifiers: Set[str], artifacts: Dict) -> Dict:
        severity_terms = ["critical", "high", "error", "warning", "failed", "denied", "blocked"]
        lower_text = chunk_text.lower()

        severity_hits = [term for term in severity_terms if term in lower_text]

        return {
            "source_type": self.name,
            "identifiers": sorted(list(identifiers)),
            "has_anomaly": bool(severity_hits),
            "source_metadata": {
                "primary_identifier_type": "ip/user/host",
                "severity_terms_found": severity_hits,
                "note": "Boilerplate Splunk parser. Adjust field extraction for actual Splunk export fields.",
            },
        }


class Rapid7Parser(BaseLogParser):
    """
    Boilerplate parser for Rapid7-style vulnerability or detection logs.

    Adjust this based on whether the export comes from InsightVM, InsightIDR, or another Rapid7 product.
    Common fields may include asset, ip, vulnerability, severity, CVE, hostname, user, alert name.
    """

    name = "rapid7"

    IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    CVE_REGEX = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
    SEVERITY_REGEX = re.compile(r"(critical|high|medium|low)", re.IGNORECASE)

    def extract_ids_from_line(self, line: str) -> Set[str]:
        ids = set()
        ids.update(self.IP_REGEX.findall(line))
        ids.update([cve.upper() for cve in self.CVE_REGEX.findall(line)])
        return ids

    def enrich_chunk(self, chunk_text: str, identifiers: Set[str], artifacts: Dict) -> Dict:
        severities = self.SEVERITY_REGEX.findall(chunk_text)

        return {
            "source_type": self.name,
            "identifiers": sorted(list(identifiers)),
            "has_anomaly": any(s.lower() in ["critical", "high"] for s in severities),
            "source_metadata": {
                "primary_identifier_type": "ip/cve/asset",
                "severities_found": list(set([s.lower() for s in severities])),
                "note": "Boilerplate Rapid7 parser. Adjust fields based on actual Rapid7 export format.",
            },
        }


class RadarParser(BaseLogParser):
    """
    Boilerplate parser for radar or sensor logs.

    Adjust field extraction based on actual radar format.
    Possible fields include sensor_id, track_id, object_id, range, velocity, azimuth, timestamp, confidence.
    """

    name = "radar"

    SENSOR_REGEX = re.compile(r"(?:sensor|sensor_id)=([A-Za-z0-9_.\-]+)", re.IGNORECASE)
    TRACK_REGEX = re.compile(r"(?:track|track_id|object_id)=([A-Za-z0-9_.\-]+)", re.IGNORECASE)

    def extract_ids_from_line(self, line: str) -> Set[str]:
        ids = set()
        ids.update(self.SENSOR_REGEX.findall(line))
        ids.update(self.TRACK_REGEX.findall(line))
        return ids

    def enrich_chunk(self, chunk_text: str, identifiers: Set[str], artifacts: Dict) -> Dict:
        lower_text = chunk_text.lower()
        anomaly_terms = ["anomaly", "lost track", "drop", "error", "interference", "unknown", "low confidence"]

        hits = [term for term in anomaly_terms if term in lower_text]

        return {
            "source_type": self.name,
            "identifiers": sorted(list(identifiers)),
            "has_anomaly": bool(hits),
            "source_metadata": {
                "primary_identifier_type": "sensor_id/track_id",
                "radar_terms_found": hits,
                "note": "Boilerplate radar parser. Adjust extraction for actual radar log schema.",
            },
        }


def get_parser(log_type: str) -> BaseLogParser:
    parsers = {
        "hdfs": HDFSParser,
        "splunk": SplunkParser,
        "rapid7": Rapid7Parser,
        "radar": RadarParser,
    }

    log_type = log_type.lower()

    if log_type not in parsers:
        raise ValueError(
            f"Unsupported log type: {log_type}. "
            f"Supported types: {', '.join(parsers.keys())}"
        )

    return parsers[log_type]()