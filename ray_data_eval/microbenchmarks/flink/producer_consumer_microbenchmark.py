import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext, ProcessFunction
import logging
import json
import resource

NUM_CPUS = 8
PRODUCER_PARALLELISM = None
CONSUMER_PARALLELISM = None
EXECUTION_MODE = "process"
MB = 1024 * 1024
GB = 1024 * MB

NUM_TASKS = 16 * 5
BLOCK_SIZE = int(1 * MB)
TIME_UNIT = 0.5

NUM_ROWS_PER_PRODUCER = 1000
NUM_ROWS_PER_CONSUMER = 100
MEMORY_USAGE_CURRENT_PROGRAM = 4 * GB


def configure_flink_memory(env: StreamExecutionEnvironment, config_path: str):
    config = Configuration()
    config.load_yaml_file(config_path)
    env.set_configuration(config)


def limit_memory(max_mem):
    """Set a memory limit in bytes."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))


def busy_loop_for_seconds(time_diff):
    start = time.perf_counter()
    i = 0
    while time.perf_counter() - start < time_diff:
        i += 1
        continue


class Producer(FlatMapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def flat_map(self, value):
        producer_start = time.time()
        # print("Producer", value)
        busy_loop_for_seconds(TIME_UNIT * 10)
        producer_end = time.time()
        log = {
            "cat": "producer:" + str(self.task_index),
            "name": "producer:" + str(self.task_index),
            "pid": "",  # Be overwritten by parse.py
            "tid": "",  # Be overwritten by parse.py
            "ts": f"{producer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{producer_end * 1e6 - producer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        logging.warning(json.dumps(log))

        for _ in range(NUM_ROWS_PER_PRODUCER):
            yield b"1" * BLOCK_SIZE


# class Consumer(MapFunction):
#     def map(self, items):
#         print("Consumer", len(items))
#         busy_loop_for_seconds(TIME_UNIT / NUM_ROWS_PER_CONSUMER)
#         return len(items)


class ConsumerActor(ProcessFunction):
    current_batch = []
    idx = 0

    def open(self, runtime_context):
        self.current_batch = []
        self.idx = 0
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def process_element(self, value, _runtime_context):
        if len(self.current_batch) == 0:
            self.consumer_start = time.time()

        # print("Consumer", self.idx, len(self.current_batch), len(value))
        busy_loop_for_seconds(TIME_UNIT / NUM_ROWS_PER_CONSUMER)
        self.current_batch.append(value)
        self.idx += 1
        if len(self.current_batch) < NUM_ROWS_PER_CONSUMER:
            return []

        self.consumer_end = time.time()
        log = {
            "cat": "consumer:" + str(self.task_index),
            "name": "consumer:" + str(self.task_index),
            "pid": "",  # Be overwritten by parse.py
            "tid": "",  # Be overwritten by parse.py
            "ts": f"{self.consumer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{self.consumer_end * 1e6 - self.consumer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        logging.warning(json.dumps(log))
        current_batch_len = len(self.current_batch)
        self.current_batch = []
        return [current_batch_len]


def run_flink(env):
    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = env.from_collection(items, type_info=Types.INT())

    producer = Producer()
    consumer = ConsumerActor()

    ds = ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY())
    if PRODUCER_PARALLELISM is not None:
        ds = ds.set_parallelism(PRODUCER_PARALLELISM)
    ds = ds.process(consumer)
    if CONSUMER_PARALLELISM is not None:
        ds = ds.set_parallelism(CONSUMER_PARALLELISM)

    result = []
    for length in ds.execute_and_collect():
        result.append(length)
        print(f"Processed block of size: {length}, {sum(result)}")

    total_length = sum(result)

    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)

    # Lower the memory so that it might possibly fit under 2GB
    # config.set_string("taskmanager.memory.process.size", "1600m")
    # config.set_string("taskmanager.memory.flink.size", "1200m")
    # config.set_string("taskmanager.memory.jvm-overhead.size", "200m")
    # config.set_string("taskmanager.memory.managed.size", "100m")
    # config.set_string("taskmanager.memory.network.size", "100m")

    # Adaptive parallelism
    if PRODUCER_PARALLELISM is None and CONSUMER_PARALLELISM is None:
        config.set_string("jobmanager.scheduler", "Adaptive")
        config.set_integer("jobmanager.adaptive-scheduler.max-parallelism-increase", NUM_CPUS)
        config.set_integer("jobmanager.adaptive-scheduler.initial-parallelism", 1)
        config.set_integer("jobmanager.adaptive-scheduler.min-parallelism", 1)
        config.set_integer("jobmanager.adaptive-scheduler.max-parallelism", NUM_CPUS)

    env = StreamExecutionEnvironment.get_execution_environment(config)

    # limit_memory(MEMORY_USAGE_CURRENT_PROGRAM)
    run_flink(env)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
