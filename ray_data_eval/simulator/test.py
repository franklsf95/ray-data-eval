import logging

from ray_data_eval.common.pipeline import ( # noqa F401
    SchedulingProblem,
    producer_consumer_problem,
    multi_stage_problem,
)
from ray_data_eval.simulator.environment import ExecutionEnvironment
from ray_data_eval.simulator.policies import (  # noqa F401
    GreedySchedulingPolicy,
    GreedyWithBufferSchedulingPolicy,
    GreedyAndAnticipatingSchedulingPolicy,
    SchedulingPolicy,
    RatesEqualizingSchedulingPolicy,
)

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
        logging.info("-" * 60)
        logging.info(f"Tick {tick}")
        env.tick()

    env.print_timeline()
    ret = env.check_all_tasks_finished()
    logging.info(f"All tasks finished? {ret}")


def main():
    problem = multi_stage_problem
    # policy = GreedySchedulingPolicy(problem)
    # policy = GreedyWithBufferSchedulingPolicy(problem)
    # policy = GreedyAndAnticipatingSchedulingPolicy(problem)
    policy = RatesEqualizingSchedulingPolicy(problem)
    test_scheduling_policy(problem, policy)


if __name__ == "__main__":
    main()
