import time
import numpy as np
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_BASIS = 0.1  # How many seconds should time_factor=1 take


def memory_blowup(x, time_factor):
    data = b"1" * DATA_SIZE_BYTES
    time.sleep(TIME_BASIS * time_factor)
    return (data, x)


def memory_shrink(item, time_factor):
    data, x = item
    time.sleep(TIME_BASIS * time_factor)
    return len(data)


def run_experiment(
    env,
    parallelism: int = -1,
    num_parts: int = 100,
    producer_time: int = 1,
    consumer_time: int = 1,
):
    start = time.perf_counter()

    items = list(range(num_parts))
    ds = env.from_collection(items, type_info=Types.INT())

    # Ways to disable chaining of two map operators
    # 1. append .disable_chaining()
    ds = ds.map(
        lambda x: memory_blowup(x, producer_time),
        output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.INT()]),
    ).disable_chaining()

    # Default, with chaining
    # ds = ds.map(
    #     lambda x: memory_blowup(x, producer_time),
    #     output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.INT()]),
    # )

    ds = ds.map(lambda x: memory_shrink(x, consumer_time), output_type=Types.LONG())

    result = ds.execute_and_collect()
    total_length = sum(result)

    end = time.perf_counter()

    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def main():
    # Using `THREAD` mode
    # config = Configuration()
    # config.set_string("python.execution-mode", "thread")
    # env = StreamExecutionEnvironment.get_execution_environment(config)

    # Using default `PROCESS` mode
    env = StreamExecutionEnvironment.get_execution_environment()

    config = {
        "parallelism": 20,
        "total_data_size_gb": 100,
        "num_parts": 100,
        "producer_time": 1,
        "consumer_time": 9,
    }
    env.set_parallelism(config["parallelism"])

    run_experiment(
        env,
        config["parallelism"],
        config["num_parts"],
        config["producer_time"],
        config["consumer_time"],
    )

    config["total_data_size"] = config["total_data_size_gb"] * 10**9
    config["num_parts"] = config["total_data_size"] // DATA_SIZE_BYTES
    config["producer_consumer_ratio"] = (
        config["producer_time"] / config["consumer_time"]
    )


if __name__ == "__main__":
    main()
