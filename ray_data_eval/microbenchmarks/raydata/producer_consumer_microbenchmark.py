import os
import time

import numpy as np

import ray


def bench():
    os.environ["RAY_DATA_OP_RESERVATION_RATIO"] = "0"

    MB = 1024 * 1024
    NUM_CPUS = 8
    NUM_TASKS = 16 * 5
    TIME_UNIT = 0.5
    BLOCK_SIZE = 1 * MB

    NUM_ROWS_PER_PRODUCER = 1000
    NUM_ROWS_PER_CONSUMER = 100

    def produce(batch):
        time.sleep(TIME_UNIT * 10)
        for id in batch["id"]:
            yield {
                "id": [id],
                "image": [np.zeros(BLOCK_SIZE, dtype=np.uint8)],
            }

    def consume(batch):
        time.sleep(TIME_UNIT)
        print("consume", batch["id"])

        return {"id": batch["id"], "result": [0 for _ in batch["id"]]}

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = MB * 100

    ray.shutdown()
    ray.init(num_cpus=NUM_CPUS)
    data_context.execution_options.resource_limits.object_store_memory = 1024 * 1024 * 1024 * 2

    ds = ray.data.range(NUM_ROWS_PER_PRODUCER * NUM_TASKS, override_num_blocks=NUM_TASKS)
    ds = ds.map_batches(produce, batch_size=NUM_ROWS_PER_PRODUCER)
    ds = ds.map_batches(consume, batch_size=NUM_ROWS_PER_CONSUMER, num_cpus=0.99)
    start_time = time.time()
    for _, _ in enumerate(ds.iter_batches(batch_size=NUM_ROWS_PER_PRODUCER)):
        pass
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    ray.timeline("timeline.json")


if __name__ == "__main__":
    bench()
