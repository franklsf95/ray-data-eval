import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
import os

NUM_CPUS = 8
MB = 1024 * 1024

NUM_TASKS = 16 * 5
TIME_UNIT = 0.5

BLOCK_SIZE = 1 * MB
NUM_ROWS_PER_PRODUCER = 1000
NUM_ROWS_PER_CONSUMER = 100


def start_spark():
    spark = (
        SparkSession.builder.appName("Local Spark Example")
        .master(f"local[{NUM_CPUS}]")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.cores.max", NUM_CPUS)
        .getOrCreate()
    )
    return spark


def producer_udf(row):
    # print('producer_udf', row.item)
    # Simulate a delay
    time.sleep(TIME_UNIT * 10)
    for j in range(NUM_ROWS_PER_PRODUCER):
        data = b"1" * BLOCK_SIZE
        yield (data, row.item * NUM_ROWS_PER_PRODUCER + j)


def consumer_udf(batch_rows):
    # print('consumer_udf', len(batch_rows))
    time.sleep(TIME_UNIT * 1 / NUM_ROWS_PER_CONSUMER)
    return (int(len(batch_rows)),)


def run_spark_data(spark):
    start = time.perf_counter()

    items = [(item,) for item in range(NUM_TASKS)]
    input_schema = StructType([StructField("item", IntegerType(), True)])
    df = spark.createDataFrame(items, schema=input_schema)

    # df = df.repartition(NUM_CPUS)
    # Applying the producer UDF
    producer_df = df.rdd.flatMap(producer_udf).toDF()
    # df = df.repartition(NUM_CPUS)
    print(producer_df.count())

    # Applying the consumer UDF
    # consumer_rdd = producer_rdd \
    #     .groupBy(lambda x: (x[1] // NUM_ROWS_PER_CONSUMER)) \
    #     .map(lambda x: consumer_udf(list(x[1])))

    result_schema = StructType([StructField("result", IntegerType(), True)])
    consumer_rdd = producer_df.rdd.map(lambda x: consumer_udf(x))
    consumer_df = consumer_rdd.toDF(result_schema)

    total_processed = consumer_df.agg({"result": "sum"}).collect()[0][0]

    run_time = time.perf_counter() - start
    print(f"\nTotal length of data processed: {total_processed:,}")
    print(f"Run time: {run_time:.2f} seconds")
    return total_processed


def run_experiment():
    spark = start_spark()
    run_spark_data(spark)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
