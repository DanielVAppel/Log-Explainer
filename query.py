#!/usr/bin/env python
import argparse
import json
import os
from typing import List, Dict

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


def load_index(index_dir: str):
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.jsonl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing index or metadata in {index_dir}. "
            "Have you run ingest.py?"
        )

    index = faiss.read_index(index_path)

    metadata: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    return index, metadata


def embed_query(query: str, model_name: str):
    model = SentenceTransformer(model_name)
    vec = model.encode([query])
    vec = vec.astype("float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve_chunks(
    query_vec,
    index,
    metadata,
    top_k: int,
    anomalies_only: bool,
) -> List[Dict]:
    distances, indices = index.search(query_vec, top_k * 2)
    # indices shape: (1, N)

    results: List[Dict] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        if anomalies_only and not chunk.get("has_anomaly", False):
            continue
        results.append(chunk)
        if len(results) >= top_k:
            break
    return results


def call_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Call a local Ollama model. Make sure 'ollama serve' is running and
    you've done `ollama pull llama3` (or another model) beforehand.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def build_prompt(question: str, chunks: List[Dict]) -> str:
    parts = [
        "You are a helpful log analysis assistant.",
        "You are given HDFS log snippets and a question.",
        "Use only the information in the snippets to answer.",
        "",
        f"Question: {question}",
        "",
        "Relevant log snippets:",
    ]

    for i, c in enumerate(chunks, start=1):
        header = (
            f"[Snippet {i}] file={c['file']} "
            f"lines={c['start_line']}-{c['end_line']} "
            f"has_anomaly={c.get('has_anomaly', False)} "
            f"block_ids={', '.join(c.get('block_ids', [])[:8])}"
        )
        parts.append(header)
        parts.append(c["text"])
        parts.append("-" * 80)

    parts.append("Answer clearly and concisely:")
    prompt = "\n".join(parts)
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Query HDFS log index with a natural language question."
    )
    parser.add_argument(
        "question",
        help="Question to ask about the logs.",
    )
    parser.add_argument(
        "--index-dir",
        default="index",
        help="Directory where the FAISS index and metadata are stored.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name (must match ingest).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3",
        help="Name of the Ollama model to use (e.g., llama3, mistral).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve.",
    )
    parser.add_argument(
        "--anomalies-only",
        action="store_true",
        help="Only use chunks that contain anomalous blocks.",
    )

    args = parser.parse_args()

    index, metadata = load_index(args.index_dir)
    qvec = embed_query(args.question, args.model_name)
    chunks = retrieve_chunks(
        qvec, index, metadata, top_k=args.top_k, anomalies_only=args.anomalies_only
    )

    if not chunks:
        print("No matching chunks found (check if index is built and data exists).")
        return

    prompt = build_prompt(args.question, chunks)
    answer = call_ollama(prompt, model=args.ollama_model)

    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== USED SNIPPETS ===")
    for i, c in enumerate(chunks, start=1):
        print(
            f"{i}. {c['file']} lines {c['start_line']}-{c['end_line']} "
            f"(has_anomaly={c.get('has_anomaly', False)})"
        )


if __name__ == "__main__":
    main()
