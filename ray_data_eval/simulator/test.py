import logging

from ray_data_eval.common.pipeline import (  # noqa F401
    SchedulingProblem,
    test_problem,
    producer_consumer_problem,
    multi_stage_problem,
    long_problem,
    training_problem,
    e2e_problem,
    e2e_problem2,
    e2e_problem3,
)
from ray_data_eval.simulator.environment import ExecutionEnvironment
from ray_data_eval.simulator.policies import (  # noqa F401
    GreedyPolicy,
    GreedyWithBufferPolicy,
    GreedyOracleProducerFirstPolicy,
    GreedyOracleConsumerFirstPolicy,
    SchedulingPolicy,
    RatesEqualizingPolicy,
    ConcurrencyCapPolicy,
    DelayPolicy,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname).1s %(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_scheduling_policy(problem: SchedulingProblem, policy: SchedulingPolicy) -> bool:
    env = ExecutionEnvironment(
        resources=problem.resources,
        buffer_size=problem.buffer_size_limit,
        tasks=problem.tasks,
        scheduling_policy=policy,
    )

    for tick in range(problem.time_limit):
        logging.info("-" * 60)
        logging.info(f"Tick {tick}")
        env.tick()
        if env.check_all_tasks_finished():
            break

    env.print_timeline()
    ret = env.check_all_tasks_finished()
    logging.info(f"All tasks finished? {ret}")


def main():
    problem = e2e_problem2
    policy = GreedyPolicy(problem)
    # policy = GreedyWithBufferPolicy(problem)
    # policy = GreedyOracleConsumerFirstPolicy(problem)
    # policy = GreedyOracleProducerFirstPolicy(problem)
    # policy = GreedyOracleConsumerFirstPolicy(problem)
    # policy = ConcurrencyCapPolicy(problem)
    # policy = RatesEqualizingPolicy(problem)
    # policy = DelayPolicy(problem)
    test_scheduling_policy(problem, policy)


if __name__ == "__main__":
    main()
