#!/usr/bin/env python
import argparse
import json
import os
import re

from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BLOCK_ID_REGEX = re.compile(r"blk_[\-\d]+")


def load_anomaly_labels(path: Optional[str]) -> Dict[str, str]:
    """
    Load anomaly_label.csv into a dict:
    { BlockId -> Label ('Normal' or 'Anomaly') }
    """
    if path is None or not os.path.exists(path):
        print("[ingest] No anomaly_label.csv found (or path missing). "
              "Continuing without labels.")
        return {}

    df = pd.read_csv(path)
    # Expected columns: BlockId,Label
    label_map = dict(zip(df["BlockId"].astype(str), df["Label"].astype(str)))
    print(f"[ingest] Loaded {len(label_map)} block labels from {path}")
    return label_map


def chunk_log_file(
    log_path: str,
    chunk_lines: int,
    labels: Dict[str, str]
) -> List[Dict]:
    """
    Stream HDFS.log, group into chunks, attach metadata:
    - file, start_line, end_line
    - block_ids (unique)
    - has_anomaly (True if any block in chunk has label 'Anomaly')
    - text (raw concatenated lines)
    """
    chunks = []
    current_lines = []
    current_start_line = 1
    current_block_ids = set()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            current_lines.append(line.rstrip("\n"))

            # extract block IDs
            for m in BLOCK_ID_REGEX.findall(line):
                current_block_ids.add(m)

            if lineno % chunk_lines == 0:
                text = "\n".join(current_lines)
                has_anomaly = any(
                    labels.get(bid, "Normal") == "Anomaly"
                    for bid in current_block_ids
                )
                chunks.append(
                    {
                        "file": os.path.basename(log_path),
                        "start_line": current_start_line,
                        "end_line": lineno,
                        "block_ids": sorted(list(current_block_ids)),
                        "has_anomaly": has_anomaly,
                        "text": text,
                    }
                )
                # reset
                current_lines = []
                current_block_ids = set()
                current_start_line = lineno + 1

        # flush remainder
        if current_lines:
            text = "\n".join(current_lines)
            has_anomaly = any(
                labels.get(bid, "Normal") == "Anomaly"
                for bid in current_block_ids
            )
            chunks.append(
                {
                    "file": os.path.basename(log_path),
                    "start_line": current_start_line,
                    "end_line": lineno,
                    "block_ids": sorted(list(current_block_ids)),
                    "has_anomaly": has_anomaly,
                    "text": text,
                }
            )

    print(f"[ingest] Created {len(chunks)} chunks from {log_path}")
    return chunks


def build_index(chunks: List[Dict], index_dir: str, model_name: str):
    os.makedirs(index_dir, exist_ok=True)

    print(f"[ingest] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]

    print("[ingest] Computing embeddings...")
    # encode in batches, convert to float32 numpy
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine

    print("[ingest] Adding vectors to FAISS index...")
    index.add(embeddings)

    # Save index and metadata
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
    parser = argparse.ArgumentParser(
        description="Ingest HDFS logs and build a vector index."
    )
    parser.add_argument(
        "--log-path",
        required=True,
        help="Path to HDFS.log (raw log file).",
    )
    parser.add_argument(
        "--anomaly-labels",
        default=None,
        help="Optional path to anomaly_label.csv.",
    )
    parser.add_argument(
        "--index-dir",
        default="index",
        help="Directory to store FAISS index and metadata.",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=200,
        help="Number of log lines per chunk.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )

    args = parser.parse_args()

    labels = load_anomaly_labels(args.anomaly_labels)
    chunks = chunk_log_file(args.log_path, args.chunk_lines, labels)
    build_index(chunks, args.index_dir, args.model_name)


if __name__ == "__main__":
    main()
