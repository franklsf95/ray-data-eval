from dataclasses import dataclass
from enum import Enum
import logging

from ray_data_eval.common.pipeline import ResourcesSpec, SchedulingProblem, TaskSpec

Resource = str
Tick = int

CPU: Resource = "CPU"
GPU: Resource = "GPU"


@dataclass
class DataItem:
    id: str
    operator_idx: int
    block_id: int = 0
    producer: TaskSpec | None = None
    produced_at: Tick = -1
    consumer: TaskSpec | None = None
    consumed_at: Tick = -1


class Buffer:
    def __init__(self, capacity: int, name: str = "buf"):
        self.capacity = capacity
        self.name = name
        self._items: list[DataItem] = []
        self._timeline: list[int] = [0]

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return f"{self}({len(self._items)}/{self.capacity})"

    def tick(self):
        logging.info(f"[{self}] Tick")
        self._timeline.append(len(self._items))

    def get_available_space(self) -> int:
        return self.capacity - len(self._items)

    def push(self, at_tick: Tick, task: TaskSpec, size: int) -> DataItem | None:
        assert size > 0, (task, size)
        if len(self._items) + size > self.capacity:
            logging.debug(f"[{self}] Cannot push {task.id}: buffer full")
            return None
        for i in range(size):
            item = DataItem(
                id=task.id,
                operator_idx=task.operator_idx,
                block_id=i,
                producer=task,
                produced_at=at_tick,
                consumer=None,
                consumed_at=-1,
            )
            self._items.append(item)
        logging.debug(f"[{self}] Pushed {task.id}")
        return item

    def remove(self, items: list[DataItem]) -> list[DataItem]:
        for item in items:
            self._items.remove(item)
        return items

    def peek(self, size: int, operator_idx: int) -> list[DataItem]:
        """
        Returns the first `size` items in the buffer without consumers.
        If there are fewer than `size` items, returns an empty list.
        """
        items = [
            item
            for item in self._items
            if item.consumer is None and item.operator_idx == operator_idx
        ]
        logging.debug(f"[{self}] Peeked {size} items: {items[:size]}")
        if len(items) < size:
            return []
        return items[:size]

    def print_timeline(self, max_time: int):
        print(f"|| {self}  ||", end="")
        for i, item in enumerate(self._timeline):
            if i >= max_time + 1:
                break
            print(f" {item:<3} |", end="")
        print("|")


class TaskStateType(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PENDING_OUTPUT = "PENDING_OUTPUT"
    FINISHED = "FINISHED"


class TaskState:
    state: TaskStateType = TaskStateType.PENDING
    started_at: Tick = -1
    execution_started_at: Tick = -1
    execution_finished_at: Tick = -1
    finished_at: Tick = -1


TaskStateMap = dict[TaskSpec, TaskState]


@dataclass
class RunningTask:
    spec: TaskSpec
    inputs: list[DataItem]
    started_at: Tick
    remaining_ticks: int


class HistoryEventType(Enum):
    TASK_STARTED = "TASK_STARTED"
    TASK_FINISHED = "TASK_FINISHED"


@dataclass
class HistoryEvent:
    tick: Tick
    type: HistoryEventType
    task: TaskSpec


class Executor:
    def __init__(self, id: str, resource: Resource, env: "ExecutionEnvironment"):
        self.id = id
        self.resource = resource
        self.running_task: RunningTask | None = None
        self._env = env
        self._events: list[HistoryEvent] = []
        self._timeline: list[str] = []

    def __repr__(self):
        return f"Executor#{self.id}"

    def _try_finishing_running_task(self) -> bool:
        """
        Tries to add the running task's output to the buffer and remove its inputs.

        NOTE(lsf): One big difference between the simulator and the real execution in Ray is that
        we remove the inputs from the buffer first before adding the outputs. In Ray, we would
        add the outputs first and then remove the inputs, therefore needing more buffer space.
        """
        if self.running_task is None:
            return True
        if self.running_task.spec.output_size > 0 and (
            self._env.buffer.get_available_space()
            < self.running_task.spec.output_size - self.running_task.spec.input_size
        ):
            logging.debug(f"[{self}] Cannot finish {self.running_task.spec.id}: buffer is full")
            return False
        self._env.buffer.remove(self.running_task.inputs)
        if self.running_task.spec.output_size > 0:
            item = self._env.buffer.push(
                at_tick=self.running_task.started_at,
                task=self.running_task.spec,
                size=self.running_task.spec.output_size,
            )
            if item is None:
                logging.debug(f"[{self}] Cannot finish {self.running_task.spec.id}: buffer is full")
                return False
        return True

    def _finish_running_task(self) -> RunningTask:
        """
        Sets the running task to None and returns the finished task.
        """
        logging.info(f"[{self}] Finished {self.running_task.spec.id}")
        self._events.append(
            HistoryEvent(
                tick=self.running_task.started_at,
                type=HistoryEventType.TASK_FINISHED,
                task=self.running_task.spec,
            )
        )
        ret = self.running_task
        self.running_task = None
        return ret

    def _get_timeline_item(self):
        if self.running_task is None:
            return ""
        if self.running_task.remaining_ticks <= 0:
            return "!"
        return self.running_task.spec.id

    def tick(self) -> RunningTask | None:
        """
        Advances a tick. Returns the task that finished, if any.
        """
        self._timeline.append(self._get_timeline_item())
        if self.running_task is None:
            return None
        self.running_task.remaining_ticks -= 1
        if self.running_task.remaining_ticks <= 0:
            if self._try_finishing_running_task():
                self._env.update_task_state(self.running_task.spec.id, TaskStateType.FINISHED)
                return self._finish_running_task()
            else:
                self._env.update_task_state(self.running_task.spec.id, TaskStateType.PENDING_OUTPUT)
                return None

    def start_task(self, task: TaskSpec, at_tick: Tick, inputs: list[DataItem]) -> bool:
        """
        Tries to start a task on this executor.
        Returns true if the task was started, false if it was not.
        """
        if (
            task.resources.cpu > 0
            and self.resource != CPU
            or task.resources.gpu > 0
            and self.resource != GPU
        ):
            return False
        if self.running_task is not None:
            logging.debug(
                f"[{self}] Cannot start {task.id}: {self.running_task.spec.id} is running"
            )
            return False
        self.running_task = RunningTask(
            spec=task,
            inputs=inputs,
            started_at=at_tick,
            remaining_ticks=task.duration,
        )
        for item in inputs:
            item.consumer = task
            item.consumed_at = at_tick
        logging.info(f"[{self}] Started {task}")
        self._events.append(
            HistoryEvent(
                tick=at_tick,
                type=HistoryEventType.TASK_STARTED,
                task=task,
            )
        )
        return True

    def cancel_task(self):
        self.running_task = None

    def print_timeline(self, max_time: int):
        print(f"|| {self.id:4} ||", end="")
        for i, item in enumerate(self._timeline):
            if "_" in item:
                item = item.split("_")[0]
            if i >= max_time + 1:
                break
            print(f" {item:<3} |", end="")
        print("|")


class ExecutionEnvironment:
    def __init__(
        self,
        *,
        resources: ResourcesSpec,
        buffer_size: int,
        tasks: list[TaskSpec],
        scheduling_policy: "SchedulingPolicy" = None,
    ):
        self.task_specs = {t.id: t for t in tasks}
        self.task_states = {t.id: TaskState() for t in tasks}
        self.buffer = Buffer(capacity=buffer_size)
        self.scheduling_policy = scheduling_policy
        self._current_tick = 0
        self._executors = [Executor(f"CPU{i}", CPU, self) for i in range(resources.cpu)] + [
            Executor(f"GPU{i}", GPU, self) for i in range(resources.gpu)
        ]

    def __repr__(self):
        return f"ExecutionEnvironment@{self._current_tick}"

    def _get_executors_sorted(self) -> list[Executor]:
        """
        Returns first the executors with tasks that are finishing and will
        decrease buffer usage.
        """

        def _sort_key(executor: Executor) -> tuple[int, int]:
            if executor.running_task is None:
                return 100000, 100000  # Sorted last.
            remaining_ticks = executor.running_task.remaining_ticks
            net_output_size = (
                executor.running_task.spec.output_size - executor.running_task.spec.input_size
            )
            # Net_output_size first: decreasing buffer usage.
            return net_output_size, remaining_ticks

        return sorted(self._executors, key=_sort_key)

    def update_task_state(self, tid: str, state: TaskStateType):
        self.task_states[tid].state = state
        if state == TaskStateType.RUNNING:
            self.task_states[tid].started_at = self._current_tick
        elif state == TaskStateType.PENDING_OUTPUT:
            self.task_states[tid].execution_finished_at = self._current_tick
        elif state == TaskStateType.FINISHED:
            self.task_states[tid].finished_at = self._current_tick
        if self.scheduling_policy is not None:
            self.scheduling_policy.on_task_state_change(self.task_specs[tid], self.task_states[tid])

    def tick(self):
        if self.scheduling_policy is not None:
            self.scheduling_policy.tick(self)
        logging.debug(f"[{self}] Tick")
        self._current_tick += 1
        for executor in self._get_executors_sorted():
            executor.tick()
        self.buffer.tick()

    def _get_task_inputs(self, task: TaskSpec) -> list[DataItem]:
        if task.input_size == 0:
            return True, []
        inp = self.buffer.peek(task.input_size, task.operator_idx - 1)
        if len(inp) < task.input_size:
            return False, []
        return True, inp

    def can_get_task_input(self, task: TaskSpec) -> bool:
        can_start, _ = self._get_task_inputs(task)
        return can_start

    def start_task(self, task: TaskSpec, executor_id: int) -> bool:
        can_start, inp = self._get_task_inputs(task)
        if can_start and self._executors[executor_id].start_task(task, self._current_tick, inp):
            self.update_task_state(task.id, TaskStateType.RUNNING)
            return True
        return False

    def start_task_on_any_executor(self, task: TaskSpec) -> bool:
        can_start, _ = self._get_task_inputs(task)
        if not can_start:
            return False
        for exec_id in range(len(self._executors)):
            if self.start_task(task, exec_id):
                return True
        return False

    def cancel_task(self, task: TaskSpec):
        raise NotImplementedError

    def print_timeline(self):
        max_time = self._current_tick
        separator_line = "++" + "-" * (max_time * 6 + 7) + "++"
        print(separator_line)
        for executor in self._executors:
            executor.print_timeline(max_time)
        print(separator_line)
        self.buffer.print_timeline(max_time)
        print(separator_line)
        print("|| time ||", end="")
        for t in range(max_time):
            print(f" {t:<3} |", end="")
        print("|")
        print(separator_line)
        print("Total Run Time =", max([s.finished_at for s in self.task_states.values()]))

    def check_all_tasks_finished(self):
        all_finished = True
        for tid, state in self.task_states.items():
            logging.info(f"{tid}: {state.state} - {state.finished_at}")
            if state.state != TaskStateType.FINISHED:
                all_finished = False
        return all_finished


class SchedulingPolicy:
    def __init__(self, problem: SchedulingProblem):
        self.problem = problem

    def tick(self, _env: ExecutionEnvironment):
        logging.info(f"[{self}] Tick")

    def on_task_state_change(self, _task: TaskSpec, _state: TaskState):
        pass
