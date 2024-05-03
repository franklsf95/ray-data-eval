from dataclasses import InitVar, dataclass, field


@dataclass
class TaskSpec:
    id: str
    duration: int
    input_size: int = 0
    output_size: int = 0
    num_cpus: int = 1


def _get_tasks(
    producer_time: list[int],
    consumer_time: list[int],
    producer_output_size: list[int],
    consumer_input_size: list[int],
):
    tasks = []
    # Consumer tasks first.
    for i, (time, size) in enumerate(zip(consumer_time, consumer_input_size)):
        tasks.append(TaskSpec(f"C{i}", time, input_size=size))
    for i, (time, size) in enumerate(zip(producer_time, producer_output_size)):
        tasks.append(TaskSpec(f"P{i}", time, output_size=size))
    return tasks


@dataclass
class SchedulingProblem:
    name: str = "TaskScheduling"
    num_producers: int = 1
    num_consumers: int = 1
    producer_time: InitVar[int | list[int]] = 1
    consumer_time: InitVar[int | list[int]] = 1
    producer_output_size: InitVar[int | list[int]] = 1
    consumer_input_size: InitVar[int | list[int]] = 1
    num_execution_slots: int = 1
    time_limit: int = 4
    buffer_size_limit: int = 1

    tasks: list[TaskSpec] = field(init=False)

    def __post_init__(
        self, producer_time, consumer_time, producer_output_size, consumer_input_size
    ):
        self.producer_time = (
            [producer_time] * self.num_producers
            if isinstance(producer_time, int)
            else producer_time
        )
        self.consumer_time = (
            [consumer_time] * self.num_consumers
            if isinstance(consumer_time, int)
            else consumer_time
        )
        self.producer_output_size = (
            [producer_output_size] * self.num_producers
            if isinstance(producer_output_size, int)
            else producer_output_size
        )
        self.consumer_input_size = (
            [consumer_input_size] * self.num_consumers
            if isinstance(consumer_input_size, int)
            else consumer_input_size
        )
        assert len(self.producer_time) == self.num_producers, (
            self.producer_time,
            self.num_producers,
        )
        assert len(self.consumer_time) == self.num_consumers, (
            self.consumer_time,
            self.num_consumers,
        )
        assert len(self.producer_output_size) == self.num_producers, (
            self.producer_output_size,
            self.num_producers,
        )
        assert len(self.consumer_input_size) == self.num_consumers, (
            self.consumer_input_size,
            self.num_consumers,
        )
        assert sum(self.producer_output_size) == sum(self.consumer_input_size), (
            producer_output_size,
            consumer_input_size,
        )

        self.num_total_tasks = self.num_producers + self.num_consumers
        self.task_time = self.producer_time + self.consumer_time
        self.tasks = _get_tasks(
            self.producer_time,
            self.consumer_time,
            self.producer_output_size,
            self.consumer_input_size,
        )


test_problem = SchedulingProblem(
    num_producers=2,
    num_consumers=2,
    producer_output_size=1,
    producer_time=1,
    consumer_time=1,
    time_limit=160,
    num_execution_slots=4,
    buffer_size_limit=8,
)
