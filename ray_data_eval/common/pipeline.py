from dataclasses import dataclass, field


@dataclass
class ResourcesSpec:
    cpu: int = 0
    gpu: int = 0
    num_executors: int = field(init=False)

    def __post_init__(self):
        self.num_executors = self.cpu + self.gpu


@dataclass
class TaskSpec:
    id: str
    operator_idx: int
    duration: int
    input_size: int
    output_size: int
    resources: ResourcesSpec


@dataclass
class OperatorSpec:
    name: str
    operator_idx: int
    num_tasks: int
    duration: int
    input_size: int
    output_size: int
    resources: ResourcesSpec
    tasks: list[TaskSpec] = field(init=False)

    def __post_init__(self):
        self.tasks = [
            TaskSpec(
                self.name + "_" + str(i),
                self.operator_idx,
                self.duration,
                self.input_size,
                self.output_size,
                self.resources,
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
    resources: ResourcesSpec
    time_limit: int
    buffer_size_limit: int
    num_operators: int = field(init=False)
    tasks: list[TaskSpec] = field(init=False)
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
                resources=ResourcesSpec(cpu=1),
            ),
            OperatorSpec(
                name="C",
                operator_idx=1,
                num_tasks=num_consumers,
                duration=consumer_time,
                input_size=consumer_input_size,
                output_size=0,
                resources=ResourcesSpec(cpu=1),
            ),
        ],
        name,
        ResourcesSpec(cpu=num_execution_slots),
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
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=0,
            resources=ResourcesSpec(cpu=1),
        ),
    ],
    name="test_problem",
    resources=ResourcesSpec(cpu=2),
    time_limit=20,
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
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="B",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=2,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=2,
            num_tasks=4,
            duration=1,
            input_size=4,
            output_size=10,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="D",
            operator_idx=3,
            num_tasks=2,
            duration=2,
            input_size=20,
            output_size=0,
            resources=ResourcesSpec(cpu=1),
        ),
    ],
    name="multi_stage_problem",
    resources=ResourcesSpec(cpu=4),
    time_limit=15,
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
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=10,
            duration=2,
            input_size=1,
            output_size=0,
            resources=ResourcesSpec(cpu=1),
        ),
    ],
    name="producer_consumer_problem",
    time_limit=15,
    buffer_size_limit=20,
    resources=ResourcesSpec(cpu=3),
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
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="B",
            operator_idx=1,
            num_tasks=50,
            duration=2,
            input_size=1,
            output_size=2,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=2,
            num_tasks=25,
            duration=1,
            input_size=4,
            output_size=0,
            resources=ResourcesSpec(cpu=1),
        ),
    ],
    name="long_problem",
    time_limit=300,
    buffer_size_limit=5000,
    resources=ResourcesSpec(cpu=3),
)

training_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=8,
            duration=1,
            input_size=0,
            output_size=1,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=1,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="T",
            operator_idx=2,
            num_tasks=4,
            duration=1,
            input_size=2,
            output_size=0,
            resources=ResourcesSpec(gpu=1),
        ),
    ],
    name="training_problem",
    resources=ResourcesSpec(cpu=3, gpu=1),
    time_limit=12,
    buffer_size_limit=4,
)


e2e_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=16,
            duration=4,
            input_size=0,
            output_size=4,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=64,
            duration=1,
            input_size=1,
            output_size=1,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="T",
            operator_idx=2,
            num_tasks=16,
            duration=1,
            input_size=10,
            output_size=0,
            resources=ResourcesSpec(gpu=1),
        ),
    ],
    name="e2e_problem",
    resources=ResourcesSpec(cpu=8, gpu=1),
    time_limit=500,  # To make tasks finish.
    buffer_size_limit=8,
)

e2e_problem2 = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=16,
            duration=4,
            input_size=0,
            output_size=4,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=64,
            duration=1,
            input_size=1,
            output_size=0,
            resources=ResourcesSpec(cpu=1),
        ),
    ],
    name="e2e_problem2",
    resources=ResourcesSpec(cpu=4),
    time_limit=100,  # To make tasks finish.
    buffer_size_limit=80,
)

e2e_problem3 = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=16,
            duration=10,
            input_size=0,
            output_size=10,
            resources=ResourcesSpec(cpu=1),
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=160,
            duration=1,
            input_size=1,
            output_size=0,
            resources=ResourcesSpec(cpu=1),
        ),
    ],
    name="e2e_problem3",
    resources=ResourcesSpec(cpu=8),
    time_limit=100,  # To make tasks finish.
    buffer_size_limit=100,
)

problems = [
    test_problem,
    multi_stage_problem,
    producer_consumer_problem,
    long_problem,
    training_problem,
    e2e_problem,
    e2e_problem2,
    e2e_problem3,
]
