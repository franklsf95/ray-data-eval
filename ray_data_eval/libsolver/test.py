import libsolver

from ray_data_eval.common.pipeline import e2e_problem3 as problem
import time

start_time = time.time()
libsolver.solve(problem)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
