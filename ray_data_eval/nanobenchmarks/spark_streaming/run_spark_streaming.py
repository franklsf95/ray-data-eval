"""
A script to generate data for spark streaming run

Runbook:
1. run Netcat as a data server: ncat --keep-open --listen -p 1234
2. in a different terminal, start the spark streaming application:
    ./bin/spark-submit path/to/file/spark_streaming_wordcount.py localhost 1234
3. run gen_streaming_data to send data
"""

import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StringType, StructField, IntegerType

from ray_data_eval.common.types import SchedulingProblem

MB = 1000 * 1000
DATA_SIZE_BYTES = 100 * MB  # 100 MB
TIME_UNIT = 1  # seconds


def start_spark(cfg):
    spark = (
        SparkSession.builder.appName("SparkStreamingRun")
        .config("spark.master", "spark://ec2-35-85-195-144.us-west-2.compute.amazonaws.com:7077")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.executor.memory", "20g")
        .config("spark.driver.memory", "2g")
        # .config("spark.driver.maxResultSize", "25g")
        .config("spark.cores.max", cfg.num_execution_slots)
        .config("spark.default.parallelism", cfg.num_producers)
        .getOrCreate()
    )
    return spark


def producer_udf(item, producer_output_size):
    i = item
    # data = "1" * (DATA_SIZE_BYTES * producer_output_size[i])
    data = "1" * (DATA_SIZE_BYTES)
    time.sleep(TIME_UNIT)
    return data, i


def consumer_udf(item, consumer_time):
    data, i = item
    # time.sleep(TIME_UNIT * consumer_time[i])
    time.sleep(TIME_UNIT)
    return len(data)


def run_spark_data(spark, cfg):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")

    # # Create a streaming DataFrame (from socket source)
    # streaming_df = (
    #     spark.readStream.format("socket").option("host", "localhost").option("port", 1234).load()
    # )
    # # Rename column
    # streaming_df = streaming_df.withColumn("item", streaming_df["value"].cast(IntegerType()))

    # Create a streaming DataFrame (from rate source: each output row contains a timestamp and value)
    streaming_df = (
        spark.readStream.format("rate")
        .option("rowsPerSecond", 1)
        # .option("numPartitions", cfg.num_producers)
        .load()
    )
    df1 = streaming_df.withColumn("item", streaming_df["value"].cast(IntegerType()))
    df1.printSchema()

    # Define UDFs
    producer_output_schema = StructType(
        [
            StructField("produced_data", StringType(), True),
            StructField("item", IntegerType(), True),
        ]
    )
    producer_udf_spark = udf(
        lambda item: producer_udf(item, cfg.producer_output_size),
        producer_output_schema,
    )

    consumer_output_schema = IntegerType()
    consumer_udf_spark = udf(
        lambda item: consumer_udf(item, cfg.consumer_time), consumer_output_schema
    )

    # Apply UDFs
    df2 = df1.withColumn("producer_result", producer_udf_spark(col("item")))
    df3 = df2.withColumn("data_length", consumer_udf_spark(col("producer_result")))

    df3.printSchema()

    # Start the streaming query
    query = df3.writeStream.outputMode("append").format("console").start()

    query.awaitTermination()
    spark.stop()


def run_experiment(cfg):
    spark = start_spark(cfg)
    run_spark_data(spark, cfg)


def main():
    run_experiment(
        SchedulingProblem(
            num_producers=8,
            num_consumers=8,
            producer_time=1,
            consumer_time=2,
            time_limit=12,
            num_execution_slots=4,
            buffer_size_limit=2,
        )
    )


if __name__ == "__main__":
    main()
