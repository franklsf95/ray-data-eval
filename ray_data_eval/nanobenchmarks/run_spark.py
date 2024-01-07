import time
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)
from ray_data_eval.common.types import SchedulingProblem

MB = 1000 * 1000
DATA_SIZE_BYTES = 100 * MB  # 100 MB
TIME_UNIT = 1  # seconds


def start_spark(cfg):
    memory_limit_mb = (cfg.buffer_size_limit * DATA_SIZE_BYTES * 9) // MB
    memory_limit = str(memory_limit_mb) + "m"

    spark = (
        SparkSession.builder.appName("Spark")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.executor.memory", memory_limit)
        .config("spark.driver.memory", memory_limit)
        .config("spark.executor.instances", cfg.num_execution_slots)
        .getOrCreate()
    )
    return spark


def producer_udf(row, cfg):
    i = row["item"]
    data = b"1" * (DATA_SIZE_BYTES * cfg.producer_output_size[i])

    time.sleep(TIME_UNIT * cfg.producer_time[i])
    return {"data": data, "idx": i}


def consumer_udf(row, cfg):
    data = row["data"]
    time.sleep(TIME_UNIT * cfg.consumer_time[row["idx"]])
    return (int(len(data)),)


def run_spark_data(spark, cfg):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")
    start = time.perf_counter()

    items = [(item,) for item in range(cfg.num_producers)]
    input_schema = ["item"]
    df = spark.createDataFrame(data=items, schema=input_schema)
    df = df.repartition(cfg.num_producers)

    df = df.rdd.map(lambda row: producer_udf(row, cfg)).toDF()

    df = df.repartition(cfg.num_producers)

    result_schema = StructType([StructField("result", IntegerType(), True)])
    df = df.rdd.map(lambda row: consumer_udf(row, cfg)).toDF(result_schema)

    ret = df.agg({"result": "sum"}).collect()[0][0]

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(df.explain())
    print(run_time)
    return ret


def run_experiment(cfg):
    spark = start_spark(cfg)
    run_spark_data(spark, cfg)


def main():
    run_experiment(
        SchedulingProblem(
            num_producers=5,
            num_consumers=5,
            # producer_time=3,
            consumer_time=2,
            # producer_output_size=2,
            # consumer_input_size=2,
            time_limit=20,
            num_execution_slots=1,
            buffer_size_limit=2,
        ),
    )


if __name__ == "__main__":
    main()
