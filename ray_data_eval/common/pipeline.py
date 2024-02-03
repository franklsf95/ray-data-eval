from dataclasses import dataclass


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


def _get_tasks(operator_list: list[OperatorSpec]):
    tasks = []
    # Reversed so that consumer first.
    for idx, operator in enumerate(reversed(operator_list)):
        tasks.extend(
            [
                TaskSpec(
                    f"{operator.name}{i}",
                    operator.operator_idx,
                    operator.duration,
                    operator.input_size,
                    operator.output_size,
                    operator.num_cpus,
                )
                for i in range(operator.num_tasks)
            ]
        )
    return tasks


@dataclass
class SchedulingProblem:
    operator_list: list[OperatorSpec]
    num_execution_slots: int = 1
    time_limit: int = 4
    buffer_size_limit: int = 1

    def __post_init__(self):
        self.tasks = _get_tasks(self.operator_list)


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
    time_limit=12,
    num_execution_slots=4,
    buffer_size_limit=100,
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
    time_limit=15,
    num_execution_slots=4,
    buffer_size_limit=100,
)