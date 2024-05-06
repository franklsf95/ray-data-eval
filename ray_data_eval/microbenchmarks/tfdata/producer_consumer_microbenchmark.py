import tensorflow as tf
import time
import os
import numpy as np

TF_PROFILER_LOGS = "logs/tf"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def busy_loop_for_seconds(time_diff):
    start = time.time()
    i = 0
    while time.time() - start < time_diff:
        i += 1
        continue


def bench():
    MB = 1024 * 1024
    NUM_TASKS = 16 * 5
    TIME_UNIT = 0.5

    BLOCK_SIZE = 1 * MB
    NUM_ROWS_PER_PRODUCER = 1000
    NUM_ROWS_PER_CONSUMER = 100

    options = tf.data.Options()
    options.autotune.enabled = True
    options.autotune.cpu_budget = 8

    def producer_fn(idx):
        busy_loop_for_seconds(10 * TIME_UNIT)
        for i in range(NUM_ROWS_PER_PRODUCER):
            data = {
                "idx": idx * NUM_ROWS_PER_PRODUCER + i,
                "data": np.full(BLOCK_SIZE, i, dtype=np.uint8),
            }
            yield data

    # def consumer_fn(idxs, datas):
    #     busy_loop_for_seconds(TIME_UNIT)
    #     print(len(datas))
    #     return len(datas)

    def consumer_fn(idxs, datas):
        busy_loop_for_seconds(TIME_UNIT)
        print(len(datas))
        return len(datas)

    start = time.perf_counter()

    items = list(range(NUM_TASKS - 1))
    ds = tf.data.Dataset.from_tensor_slices(items)
    ds = ds.with_options(options).interleave(
        lambda item: tf.data.Dataset.from_generator(
            producer_fn,
            args=(item,),
            output_signature={
                "idx": tf.TensorSpec(shape=(), dtype=tf.int64),
                "data": tf.TensorSpec(shape=(BLOCK_SIZE,), dtype=tf.uint8),
            },
            name="producer",
        ),
        # cycle_length=1,
        block_length=NUM_ROWS_PER_CONSUMER,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="producer_interleave",
    )

    if NUM_ROWS_PER_CONSUMER > 1:
        ds = ds.batch(NUM_ROWS_PER_CONSUMER)  # Group items into batches

    ds = ds.with_options(options).map(
        lambda items: tf.numpy_function(
            consumer_fn,
            inp=[items["idx"], items["data"]],  # Process batches of items
            Tout=tf.int64,
            name="consumer",
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="consumer_map",
    )

    ret = 0
    for row in ds:
        ret += row
    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(f"Run time: {run_time:.2f} seconds")


if __name__ == "__main__":
    if not os.path.exists(TF_PROFILER_LOGS):
        os.makedirs(TF_PROFILER_LOGS)
    # tf.profiler.experimental.start(TF_PROFILER_LOGS)
    bench()
    # tf.profiler.experimental.stop()
