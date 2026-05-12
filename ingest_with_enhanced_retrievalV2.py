#!/usr/bin/env python
# Enhancement note:
'''This version improves retrieval by building FAISS embeddings from an enriched text representation of each chunk instead of using only the raw log lines. In version 1, structured metadata (anomaly labels, event traces, event occurrence summaries, and event templates) was only added at prompt time, which helped the LLM explain retrieved chunks but did not help FAISS retrieve better chunks in the first place. By incorporating compact structured metadata into the embedding text, retrieval can better match queries that depend on event patterns, anomaly context, and template-level information rather than only the wording of raw log lines.'''

import argparse
import json
import os
import re
from typing import List, Dict, Optional
from source_parsers import get_parser
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
    parser,
    artifacts: Dict,
) -> List[Dict]:
    chunks = []
    current_lines = []
    current_start_line = 1
    current_identifiers = set()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            clean_line = line.rstrip("\n")
            current_lines.append(clean_line)

            current_identifiers.update(parser.extract_ids_from_line(clean_line))

            if lineno % chunk_lines == 0:
                chunks.append(
                    build_chunk(
                        log_path=log_path,
                        start_line=current_start_line,
                        end_line=lineno,
                        lines=current_lines,
                        identifiers=current_identifiers,
                        parser=parser,
                        artifacts=artifacts,
                    )
                )

                current_lines = []
                current_identifiers = set()
                current_start_line = lineno + 1

        if current_lines:
            chunks.append(
                build_chunk(
                    log_path=log_path,
                    start_line=current_start_line,
                    end_line=lineno,
                    lines=current_lines,
                    identifiers=current_identifiers,
                    parser=parser,
                    artifacts=artifacts,
                )
            )

    print(f"[ingest] Created {len(chunks)} chunks from {log_path}")
    return chunks


def build_chunk(
    log_path: str,
    start_line: int,
    end_line: int,
    lines: List[str],
    identifiers: set,
    parser,
    artifacts: Dict,
) -> Dict:
    text = "\n".join(lines)

    parser_metadata = parser.enrich_chunk(
        chunk_text=text,
        identifiers=identifiers,
        artifacts=artifacts,
    )

    chunk = {
        "file": os.path.basename(log_path),
        "start_line": start_line,
        "end_line": end_line,
        "text": text,
    }

    chunk.update(parser_metadata)

    return chunk

def build_embedding_text(chunk: Dict) -> str:
    """
    Build an enriched text representation for embedding.
    Works for HDFS and future log types.
    """
    parts = []

    parts.append(f"source_type={chunk.get('source_type', 'unknown')}")
    parts.append(f"file={chunk['file']}")
    parts.append(f"lines={chunk['start_line']}-{chunk['end_line']}")
    parts.append(f"has_anomaly={chunk.get('has_anomaly', False)}")

    identifiers = chunk.get("identifiers", [])
    if identifiers:
        parts.append("identifiers: " + ", ".join(identifiers[:10]))

    block_ids = chunk.get("block_ids", [])
    if block_ids:
        parts.append("block_ids: " + ", ".join(block_ids[:8]))

    source_metadata = chunk.get("source_metadata", {})
    if source_metadata:
        parts.append("source metadata:")
        for key, value in source_metadata.items():
            parts.append(f"{key}: {str(value)[:300]}")

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
    parser = argparse.ArgumentParser(description="Ingest logs and build a vector index.")

    parser.add_argument(
        "--log-type",
        default="hdfs",
        choices=["hdfs", "splunk", "rapid7", "radar"],
        help="Type of logs being ingested.",
    )

    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Optional directory containing metadata/artifact files for the selected log type.",
    )

    parser.add_argument("--log-path", required=True, help="Path to log file")
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

    selected_parser = get_parser(args.log_type)

    artifacts = selected_parser.load_artifacts(
        metadata_dir=args.metadata_dir,
        anomaly_labels=args.anomaly_labels,
        event_traces=args.event_traces,
        event_occurrence=args.event_occurrence,
        log_templates=args.log_templates,
    )

    chunks = chunk_log_file(
        log_path=args.log_path,
        chunk_lines=args.chunk_lines,
        parser=selected_parser,
        artifacts=artifacts,
    )

    build_index(chunks, args.index_dir, args.model_name)


if __name__ == "__main__":
    main()