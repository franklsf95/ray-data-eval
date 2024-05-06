"""
A Hello World example of Spark Structured Streaming: Word count

Runbook:
1. run Netcat as a data server: nc -lk 9999
2. in a different terminal, start the example:
    ./bin/spark-submit path/to/file/spark_streaming_wordcount.py localhost 9999
3. any lines typed in the terminal running the netcat server will be counted and printed on screen every second
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split


def main():
    spark = (
        SparkSession.builder.appName("StructuredNetworkWordCount")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .getOrCreate()
    )

    # Create DataFrame representing the stream of input lines from connection to localhost:9999
    lines = (
        spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
    )

    # Split the lines into words
    words = lines.select(explode(split(lines.value, " ")).alias("word"))

    # Generate running word count
    wordCounts = words.groupBy("word").count()

    # Start running the query that prints the running counts to the console
    query = (
        wordCounts.writeStream.outputMode("complete").format("console").start()
    )  # starts the streaming computation

    # prevent the process from exiting while the query is active
    query.awaitTermination()


if __name__ == "__main__":
    main()
