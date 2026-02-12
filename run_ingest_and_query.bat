@echo off
cd /d "C:\Users\vince\OneDrive\Documents\GitHub\Log-Explainer"

REM Run ingest (build index)
python ingest.py --log-path "HDFS_v1\HDFS.log" --anomaly-labels "HDFS_v1\preprocessed\anomaly_label.csv" --index-dir "index" --chunk-lines 200

REM Example query (change the question as you like)
python query.py "Summarize what is happening with anomalous blocks." --anomalies-only

pause
