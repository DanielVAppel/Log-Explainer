# HDFS Log Explainer (RAG + Local LLM)

This Project was created to fufill my Masters Project requirement This project lets you **ask natural-language questions** about the HDFS log dataset and get answers from a local **generative model** (via [Ollama](https://ollama.com/)), using a vector index over log chunks.

It is designed for the **LogHub HDFS dataset**, but can be adapted to other log files.

## 1. Prerequisites

- Python 3.9+ (recommended)
- Git (optional)
- The HDFS_v1 dataset avaiable for download at: https://github.com/logpai/loghub/tree/master?tab=readme-ov-file
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- Any other pulled model of choice

# Run the following Commands once initially:
```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
deactivate

ollama pull llama3

python ingest_with_enhanced_retrievalV2.py --log-path "HDFS_v1\HDFS.log" --anomaly-labels "HDFS_v1\preprocessed\anomaly_label.csv" --event-traces "HDFS_v1\preprocessed\Event_traces.csv" --event-occurrence "HDFS_v1\preprocessed\Event_occurrence_matrix.csv" --log-templates "HDFS_v1\preprocessed\HDFS.log_templates.csv" --index-dir "index" --chunk-lines 200

# Smaller size if wanted
python ingest.py --log-path "HDFS_v1\HDFS.log" --anomaly-labels "HDFS_v1\preprocessed\anomaly_label.csv" --event-traces "HDFS_v1\preprocessed\Event_traces.csv" --event-occurrence "HDFS_v1\preprocessed\Event_occurrence_matrix.csv" --log-templates "HDFS_v1\preprocessed\HDFS.log_templates.csv" --index-dir "index" --chunk-lines 80

```
# (Run the following everytime you want the program to run:)
  # In a seperate terminal start the server:
- ollama serve

# (in a seperate terminal run the following one at a time to test the program:)
- `.\.venv\Scripts\activate`
- python query.py "Summarize what is going wrong with the anomalous blocks." --anomalies-only
- python query.py "Summarize what is going wrong with the anomalous blocks." --top-k 1
- python query.py "What is the main error here?" --anomalies-only --top-k 1 #take smaller chunks of data.
- python query.py "What patterns do you see around block blk_-3544583377289625738?" #replace with block in top-k results.

# --Interactive Mode--
# This mode lets you continously ask questions without reloading the model each time and is recommended when tags and search parameters do not need to be changed.
- python query.py --interactive --anomalies-only --top-k 1

#examples:
- Summarize what is going wrong with anomalous blocks.
- What events happen right before blocks get added to invalidSet?
- What malicious patterns or errors can you identify within the blocks causing them to be labeled as an anomaly?
- exit

# Other Useful commands for debugging:
- ollama list #Shows all avaiable models
- ollama --version #shows the current version and verifies ollama is installed.
- ollama run llama3 "say hi in 5 words" #Tests if model can respond
# Another Way to Run: (only reccomended for the first time)
- .\run_ingest_and_query.bat
# (in Powershell)
- where.exe ollama #finds the filepath for Ollama, useful for adding to environment PATH
