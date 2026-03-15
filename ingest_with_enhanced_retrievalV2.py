#!/usr/bin/env python
# Enhancement note:
'''This version improves retrieval by building FAISS embeddings from an enriched text representation of each chunk instead of using only the raw log lines. In version 1, structured metadata (anomaly labels, event traces, event occurrence summaries, and event templates) was only added at prompt time, which helped the LLM explain retrieved chunks but did not help FAISS retrieve better chunks in the first place. By incorporating compact structured metadata into the embedding text, retrieval can better match queries that depend on event patterns, anomaly context, and template-level information rather than only the wording of raw log lines.'''

import argparse
import json
import os
import re
from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BLOCK_ID_REGEX = re.compile(r"blk_[\-\d]+")


def load_anomaly_labels(path: Optional[str]) -> Dict[str, str]:
    if path is None or not os.path.exists(path):
        print("[ingest] No anomaly_label.csv found. Continuing without labels.")
        return {}

    df = pd.read_csv(path)
    return dict(zip(df["BlockId"].astype(str), df["Label"].astype(str)))


def load_event_traces(path: Optional[str]) -> Dict[str, Dict]:
    if path is None or not os.path.exists(path):
        print("[ingest] No Event_traces.csv found. Continuing without event traces.")
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


def load_event_occurrence_matrix(path: Optional[str]) -> Dict[str, Dict]:
    if path is None or not os.path.exists(path):
        print("[ingest] No Event_occurrence_matrix.csv found. Continuing without occurrence matrix.")
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


def load_log_templates(path: Optional[str]) -> Dict[str, str]:
    if path is None or not os.path.exists(path):
        print("[ingest] No HDFS.log_templates.csv found. Continuing without templates.")
        return {}

    df = pd.read_csv(path)
    return dict(zip(df["EventId"].astype(str), df["EventTemplate"].astype(str)))


def extract_event_ids(features_str: str) -> List[str]:
    if not features_str:
        return []
    return re.findall(r"E\d+", features_str)


def top_event_counts(event_counts: Dict[str, int], top_n: int = 5) -> Dict[str, int]:
    return dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])


def chunk_log_file(
    log_path: str,
    chunk_lines: int,
    labels: Dict[str, str],
    event_traces: Dict[str, Dict],
    event_occurrence: Dict[str, Dict],
    log_templates: Dict[str, str],
) -> List[Dict]:
    chunks = []
    current_lines = []
    current_start_line = 1
    current_block_ids = set()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            current_lines.append(line.rstrip("\n"))

            for m in BLOCK_ID_REGEX.findall(line):
                current_block_ids.add(m)

            if lineno % chunk_lines == 0:
                chunks.append(
                    build_chunk(
                        log_path,
                        current_start_line,
                        lineno,
                        current_lines,
                        current_block_ids,
                        labels,
                        event_traces,
                        event_occurrence,
                        log_templates,
                    )
                )
                current_lines = []
                current_block_ids = set()
                current_start_line = lineno + 1

        if current_lines:
            chunks.append(
                build_chunk(
                    log_path,
                    current_start_line,
                    lineno,
                    current_lines,
                    current_block_ids,
                    labels,
                    event_traces,
                    event_occurrence,
                    log_templates,
                )
            )

    print(f"[ingest] Created {len(chunks)} chunks from {log_path}")
    return chunks


def build_chunk(
    log_path: str,
    start_line: int,
    end_line: int,
    lines: List[str],
    block_ids: set,
    labels: Dict[str, str],
    event_traces: Dict[str, Dict],
    event_occurrence: Dict[str, Dict],
    log_templates: Dict[str, str],
) -> Dict:
    text = "\n".join(lines)
    block_ids_sorted = sorted(list(block_ids))

    has_anomaly = any(labels.get(bid, "Normal") == "Anomaly" for bid in block_ids_sorted)

    chunk_trace_data = {}
    chunk_occurrence_data = {}
    chunk_templates = {}
    chunk_event_ids = set()

    for bid in block_ids_sorted:
        if bid in event_traces:
            trace_info = event_traces[bid]
            chunk_trace_data[bid] = trace_info
            for event_id in extract_event_ids(trace_info.get("features", "")):
                chunk_event_ids.add(event_id)

        if bid in event_occurrence:
            occ_info = event_occurrence[bid]
            counts = occ_info.get("event_counts", {})
            chunk_occurrence_data[bid] = {
                "occurrence_label": occ_info.get("occurrence_label", ""),
                "occurrence_type": occ_info.get("occurrence_type", ""),
                "top_event_counts": top_event_counts(counts, top_n=5),
            }
            for event_id in counts.keys():
                chunk_event_ids.add(event_id)

    for event_id in sorted(chunk_event_ids):
        if event_id in log_templates:
            chunk_templates[event_id] = log_templates[event_id]

    return {
        "file": os.path.basename(log_path),
        "start_line": start_line,
        "end_line": end_line,
        "block_ids": block_ids_sorted,
        "has_anomaly": has_anomaly,
        "text": text,
        "trace_data": chunk_trace_data,
        "occurrence_data": chunk_occurrence_data,
        "event_templates": chunk_templates,
    }

def build_embedding_text(chunk: Dict) -> str:
    """
    Build an enriched text representation for embedding.
    This helps FAISS retrieval use both raw log content and compact structured metadata.
    """
    parts = []

    parts.append(f"file={chunk['file']}")
    parts.append(f"lines={chunk['start_line']}-{chunk['end_line']}")
    parts.append(f"has_anomaly={chunk.get('has_anomaly', False)}")

    block_ids = chunk.get("block_ids", [])[:8]
    if block_ids:
        parts.append("block_ids: " + ", ".join(block_ids))

    trace_data = chunk.get("trace_data", {})
    if trace_data:
        parts.append("event trace summary:")
        for i, (block_id, info) in enumerate(trace_data.items()):
            if i >= 3:
                break
            features = info.get("features", "")
            parts.append(
                f"{block_id}: label={info.get('trace_label', '')}, "
                f"type={info.get('trace_type', '')}, "
                f"features={features[:150]}"
            )

    occurrence_data = chunk.get("occurrence_data", {})
    if occurrence_data:
        parts.append("event occurrence summary:")
        for i, (block_id, info) in enumerate(occurrence_data.items()):
            if i >= 3:
                break
            parts.append(
                f"{block_id}: label={info.get('occurrence_label', '')}, "
                f"type={info.get('occurrence_type', '')}, "
                f"top_event_counts={info.get('top_event_counts', {})}"
            )

    event_templates = chunk.get("event_templates", {})
    if event_templates:
        parts.append("event templates:")
        for i, (event_id, template) in enumerate(event_templates.items()):
            if i >= 6:
                break
            parts.append(f"{event_id}: {template}")

    parts.append("raw log text:")
    parts.append(chunk["text"][:4000])

    return "\n".join(parts)

def build_index(chunks: List[Dict], index_dir: str, model_name: str):
    os.makedirs(index_dir, exist_ok=True)

    print(f"[ingest] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Build embeddings from enriched chunk text rather than raw log text only.
    texts = [build_embedding_text(c) for c in chunks]

    print("[ingest] Computing embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.jsonl")
    emb_path = os.path.join(index_dir, "chunks.npy")

    faiss.write_index(index, index_path)
    np.save(emb_path, embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"[ingest] Saved FAISS index to {index_path}")
    print(f"[ingest] Saved embeddings to {emb_path}")
    print(f"[ingest] Saved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest HDFS logs and build a vector index.")
    parser.add_argument("--log-path", required=True, help="Path to HDFS.log")
    parser.add_argument("--anomaly-labels", default=None, help="Path to anomaly_label.csv")
    parser.add_argument("--event-traces", default=None, help="Path to Event_traces.csv")
    parser.add_argument("--event-occurrence", default=None, help="Path to Event_occurrence_matrix.csv")
    parser.add_argument("--log-templates", default=None, help="Path to HDFS.log_templates.csv")
    parser.add_argument("--index-dir", default="index", help="Directory to store FAISS index")
    parser.add_argument("--chunk-lines", type=int, default=200, help="Number of log lines per chunk")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )

    args = parser.parse_args()

    labels = load_anomaly_labels(args.anomaly_labels)
    event_traces = load_event_traces(args.event_traces)
    event_occurrence = load_event_occurrence_matrix(args.event_occurrence)
    log_templates = load_log_templates(args.log_templates)

    chunks = chunk_log_file(
        args.log_path,
        args.chunk_lines,
        labels,
        event_traces,
        event_occurrence,
        log_templates,
    )

    build_index(chunks, args.index_dir, args.model_name)


if __name__ == "__main__":
    main()