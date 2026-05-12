# LogInsight: Interpreting Log Anomalies using Hybrid Retrieval and Large Language Models

LogInsight is a hybrid retrieval and local LLM system for analyzing large log datasets using natural-language questions. The system combines enriched log chunk retrieval, FAISS vector search, structured metadata enrichment, and local language model reasoning through Ollama.

The current implementation is built around the LogHub HDFS_v1 dataset but includes generalized parser support for future log sources such as Splunk, Rapid7, and radar/sensor logs.

---

# Features

- FAISS vector retrieval
- Structured metadata enrichment
- Local LLM inference using Ollama
- Interactive querying mode
- Highlighted suspicious log lines
- Event trace summaries
- Event occurrence summaries
- HDFS event template integration
- Extensible parser architecture for future log sources

---

# Prerequisites

- Python 3.9+
- Ollama installed: https://ollama.com/
- HDFS_v1 dataset from LogHub:
  https://github.com/logpai/loghub
- Git (optional)

---

# First-Time Setup

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pull the default Ollama model:

```powershell
ollama pull llama3
```

Build the HDFS index:

```powershell
python ingest_with_enhanced_retrievalV2.py --log-type hdfs --log-path "HDFS_v1\HDFS.log" --metadata-dir "HDFS_v1\preprocessed" --index-dir "index" --chunk-lines 200
```

Optional smaller chunk size:

```powershell
python ingest_with_enhanced_retrievalV2.py --log-type hdfs --log-path "HDFS_v1\HDFS.log" --metadata-dir "HDFS_v1\preprocessed" --index-dir "index" --chunk-lines 80
```

---

# Running the Program After Setup

Terminal 1:

```powershell
ollama serve
```

Terminal 2:

```powershell
.\.venv\Scripts\activate
python query.py --interactive --anomalies-only --top-k 1
```

Interactive mode keeps:
- the embedding model loaded
- the FAISS index loaded
- metadata loaded

This avoids reloading everything between questions and improves response speed.

Exit interactive mode with:

```text
exit
```

---

# Example Interactive Questions

```text
Summarize what is going wrong with anomalous blocks.
```

```text
What events happen right before blocks get added to invalidSet?
```

```text
What malicious patterns or errors can you identify within the blocks causing them to be labeled as an anomaly?
```

```text
What patterns do you see around block blk_-3544583377289625738?
```

---

# One-Shot Query Examples

```powershell
python query.py "Summarize what is going wrong with the anomalous blocks." --anomalies-only --top-k 1
```

```powershell
python query.py "What is the main error here?" --anomalies-only --top-k 1
```

```powershell
python query.py "What patterns do you see around block blk_-3544583377289625738?"
```

---

# Using Different Ollama Models

Pull another model:

```powershell
ollama pull phi3
```

Run with that model:

```powershell
python query.py --interactive --anomalies-only --top-k 1 --ollama-model phi3
```

Other compatible models may include:
- mistral
- phi3
- llama3
- gemma

Changing the Ollama model does NOT require rerunning ingestion.

Re-run ingestion only if you change:
- embedding model
- parser logic
- metadata enrichment
- chunk size
- dataset
---

# Future Log Source Examples

## Splunk

```powershell
python ingest_with_enhanced_retrievalV2.py --log-type splunk --log-path "logs\splunk_export.log" --index-dir "index_splunk" --chunk-lines 200
```

Current boilerplate support:
- IP extraction
- username extraction
- hostname extraction
- severity keyword detection

---

## Rapid7

```powershell
python ingest_with_enhanced_retrievalV2.py --log-type rapid7 --log-path "logs\rapid7_export.log" --index-dir "index_rapid7" --chunk-lines 200
```

Current boilerplate support:
- CVE extraction
- IP extraction
- severity extraction

---

## Radar / Sensor Logs

```powershell
python ingest_with_enhanced_retrievalV2.py --log-type radar --log-path "logs\radar.log" --index-dir "index_radar" --chunk-lines 200
```

Current boilerplate support:
- sensor ID extraction
- track ID extraction
- anomaly keyword detection

---

# Adding Additional Log Sources

Additional log sources can be added in:

```text
source_parsers.py
```

Each parser subclasses:

```python
BaseLogParser
```

and can implement:
- extract_ids_from_line()
- load_artifacts()
- enrich_chunk()

---

# Useful Debugging Commands

Verify Ollama installation:

```powershell
ollama --version
```

List installed models:

```powershell
ollama list
```

Quick model test:

```powershell
ollama run llama3 "say hi in 5 words"
```

Find Ollama executable path:

```powershell
where.exe ollama
```

---

# Notes

- `ingest_with_enhanced_retrievalV2.py` builds the FAISS index and metadata files.
- `query.py` performs retrieval and sends prompts to the local Ollama model.
- FAISS stores vector embeddings only.
- Structured metadata remains stored separately in `metadata.jsonl`.
- Retrieved metadata is added during prompt construction before LLM inference.
- Splunk, Rapid7, and radar parsers are starter templates and should be customized for real deployments.