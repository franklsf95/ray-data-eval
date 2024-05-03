import time
import logging
import json

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, MapFunction, RuntimeContext
from pyflink.common import Configuration

from ray_data_eval.common.types import SchedulingProblem, test_problem

DATA_SIZE_BYTES = 1000 * 1000 * 1  # 100 MB
TIME_UNIT = 1  # seconds
PRODUCER_PARALLELISM = 2
CONSUMER_PARALLELISM = 4

EXECUTION_MODE = "process"


class Producer(MapFunction):
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_info = None

    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, item):
        producer_start = time.time()
        data = b"1" * (DATA_SIZE_BYTES * self.cfg.producer_output_size[item])
        time.sleep(TIME_UNIT * self.cfg.producer_time[item])
        producer_end = time.time()

        log = {
            "cat": "producer:" + str(self.task_index),
            "name": "producer:" + str(self.task_index),
            "pid": "",
            "tid": "",
            "ts": f"{producer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{producer_end * 1e6 - producer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        logging.warning(json.dumps(log))
        return (data, item)


class Consumer(MapFunction):
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_info = None

    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, item):
        consumer_start = time.time()
        data, i = item
        time.sleep(TIME_UNIT * self.cfg.consumer_time[i])
        consumer_end = time.time()

        log = {
            "cat": "consumer:" + str(self.task_index),
            "name": "consumer:" + str(self.task_index),
            "pid": "",
            "tid": "",
            "ts": f"{consumer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{consumer_end * 1e6 - consumer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        logging.warning(json.dumps(log))
        return len(data)


def run_flink(env, cfg: SchedulingProblem):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")

    start = time.perf_counter()

    items = list(range(cfg.num_producers))
    ds = env.from_collection(items, type_info=Types.INT())

    producer = Producer(cfg)
    consumer = Consumer(cfg)

    ds = (
        ds.map(producer, output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.INT()]))
        .set_parallelism(PRODUCER_PARALLELISM)
        .disable_chaining()
    )

    ds = (
        ds.map(consumer, output_type=Types.LONG())
        .set_parallelism(CONSUMER_PARALLELISM)
        .disable_chaining()
    )

    result = ds.execute_and_collect()
    total_length = sum(result)

    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment(cfg: SchedulingProblem):
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    env = StreamExecutionEnvironment.get_execution_environment(config)

    # env.set_parallelism(cfg.num_producers)
    # env.set_parallelism(2)

    run_flink(env, cfg)


def main():
    run_experiment(test_problem)


if __name__ == "__main__":
    main()
