from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    id: str
    operator_idx: int
    duration: int
    input_size: int
    output_size: int
    num_cpus: int


@dataclass
class OperatorSpec:
    name: str
    operator_idx: int
    num_tasks: int
    duration: int
    input_size: int
    output_size: int
    num_cpus: int
    tasks: list[TaskSpec] = field(init=False)

    def __post_init__(self):
        self.tasks = [
            TaskSpec(
                f"{self.name}{i}",
                self.operator_idx,
                self.duration,
                self.input_size,
                self.output_size,
                self.num_cpus,
            )
            for i in range(self.num_tasks)
        ]


def _get_tasks(operators: list[OperatorSpec]):
    tasks = []
    # Reversed so that downstream tasks are prioritized.
    # TODO(MaoZiming): Sort in the policy.
    for _, operator in enumerate(reversed(operators)):
        tasks.extend(operator.tasks)
    return tasks


@dataclass
class SchedulingProblem:
    operators: list[OperatorSpec]
    name: str
    num_execution_slots: int
    time_limit: int
    buffer_size_limit: int
    num_operators: int = field(init=False)
    tasks: list = field(init=False)
    num_total_tasks: int = field(init=False)

    def __post_init__(self):
        self.num_operators = len(self.operators)
        self.tasks = _get_tasks(self.operators)
        self.num_total_tasks = len(self.tasks)


def make_producer_consumer_problem(
    name: str = "producer_consumer",
    num_producers: int = 1,
    num_consumers: int = 1,
    producer_time: int = 1,
    consumer_time: int = 1,
    producer_output_size: int = 1,
    consumer_input_size: int = 1,
    num_execution_slots: int = 1,
    time_limit: int = 4,
    buffer_size_limit: int = 1,
) -> SchedulingProblem:
    return SchedulingProblem(
        [
            OperatorSpec(
                name="P",
                operator_idx=0,
                num_tasks=num_producers,
                duration=producer_time,
                input_size=0,
                output_size=producer_output_size,
                num_cpus=1,
            ),
            OperatorSpec(
                name="C",
                operator_idx=1,
                num_tasks=num_consumers,
                duration=consumer_time,
                input_size=consumer_input_size,
                output_size=0,
                num_cpus=1,
            ),
        ],
        name,
        num_execution_slots,
        time_limit,
        buffer_size_limit,
    )


test_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=8,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="test_problem",
    time_limit=12,
    num_execution_slots=4,
    buffer_size_limit=4,
)

multi_stage_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="A",
            operator_idx=0,
            num_tasks=8,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="B",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=2,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=2,
            num_tasks=4,
            duration=1,
            input_size=4,
            output_size=10,
            num_cpus=1,
        ),
        OperatorSpec(
            name="D",
            operator_idx=3,
            num_tasks=2,
            duration=2,
            input_size=20,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="multi_stage_problem",
    time_limit=15,
    num_execution_slots=4,
    buffer_size_limit=100,
)

producer_consumer_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=10,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=10,
            duration=2,
            input_size=1,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="producer_consumer_problem",
    time_limit=15,
    buffer_size_limit=20,
    num_execution_slots=3,
)

long_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="A",
            operator_idx=0,
            num_tasks=50,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="B",
            operator_idx=1,
            num_tasks=50,
            duration=2,
            input_size=1,
            output_size=2,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=2,
            num_tasks=25,
            duration=1,
            input_size=4,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="long_problem",
    time_limit=300,
    buffer_size_limit=5000,
    num_execution_slots=3,
)

training_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=5,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=5,
            duration=2,
            input_size=1,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="T",
            operator_idx=2,
            num_tasks=5,
            duration=2,
            input_size=1,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="training_problem",
    time_limit=12,
    num_execution_slots=4,
    buffer_size_limit=4,
)

problems = [test_problem, multi_stage_problem, producer_consumer_problem, long_problem]
