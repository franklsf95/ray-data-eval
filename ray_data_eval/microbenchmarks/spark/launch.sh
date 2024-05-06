export SPARK_EVENTS_PATH=/home/ubuntu/ray-data-eval/logs/spark-events
export SPARK_TRACE_EVENT_PATH=/home/ubuntu/ray-data-eval/logs/spark-trace-events
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HOME=/home/ubuntu/miniconda3/envs/ray-data/lib/python3.11/site-packages/pyspark
export PYSPARK_PYTHON=/home/ubuntu/miniconda3/envs/ray-data/bin/python
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"

/home/ubuntu/miniconda3/envs/ray-data/lib/python3.11/site-packages/pyspark/sbin/stop-history-server.sh
/home/ubuntu/miniconda3/envs/ray-data/lib/python3.11/site-packages/pyspark/sbin/start-history-server.sh
python producer_consumer_microbenchmark.py