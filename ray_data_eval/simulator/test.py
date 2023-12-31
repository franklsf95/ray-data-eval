import logging

from ray_data_eval.common.types import SchedulingProblem, test_problem
from ray_data_eval.simulator.environment import ExecutionEnvironment
from ray_data_eval.simulator.policies import GreedyWithBufferSchedulingPolicy, SchedulingPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname).1s %(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_scheduling_policy(problem: SchedulingProblem, policy: SchedulingPolicy) -> bool:
    env = ExecutionEnvironment(
        num_executors=problem.num_execution_slots,
        buffer_size=problem.buffer_size_limit,
        tasks=problem.tasks,
        scheduling_policy=policy,
    )

    for tick in range(problem.time_limit):
        logging.info(f"Tick {tick}")
        env.tick()

    env.print_timeline()
    ret = env.check_all_tasks_finished()
    logging.info(f"All tasks finished? {ret}")


def main():
    problem = test_problem
    policy = GreedyWithBufferSchedulingPolicy(problem)
    test_scheduling_policy(problem, policy)


if __name__ == "__main__":
    main()
