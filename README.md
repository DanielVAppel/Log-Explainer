# Log-Explainer
This project lets you ask natural-language questions about a log dataset and get answers from a local generative model ->

# HDFS Log Explainer (RAG + Local LLM)

This project lets you **ask natural-language questions** about the
HDFS log dataset and get answers from a local **generative model** (via
[Ollama](https://ollama.com/)), using a vector index over log chunks.

It is designed for the **LogHub HDFS dataset**, but can be adapted to
other log files.

## 1. Prerequisites

- Python 3.9+ (recommended)
- Git (optional)
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- A pulled model, for example:

```bash
ollama pull llama3

python m -pip install -r requirements.txt

python ingest.py ^
  --log-path "C:/Users/vince/Downloads/HDFS_v1/HDFS.log" ^
  --anomaly-labels "C:/Users/vince/Downloads/HDFS_v1/preprocessed/anomaly_label.csv" ^
  --index-dir "index" ^
  --chunk-lines 200

or(one line)

python ingest.py --log-path "HDFS_v1\HDFS.log" --anomaly-labels "HDFS_v1\preprocessed\anomaly_label.csv" --index-dir "index" --chunk-lines 200

(in a new seperate termianl run the following:)
ollama serve

(in the previous terminal run the following:)
python query.py "Summarize what is going wrong with the anomalous blocks." --anomalies-only

python query.py "What patterns do you see around block blk_-3544583377289625738?"

# Another Way to Run:
.\run_ingest_and_query.bat