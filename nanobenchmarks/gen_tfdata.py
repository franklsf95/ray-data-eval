import time

import tensorflow as tf
import numpy as np
import wandb


DATA_SIZE = 1000 * 100


def gen_data(_):
    data = tf.random.uniform(shape=[DATA_SIZE], dtype=tf.float64) # using tf.float64 because in ray data version, np.random.rand() returns items of type np.float64
    return data 


def memory_blowup(x, blowup: int):
    print(f"memory_blowup({blowup})")
    random_arrays = [x + tf.random.uniform(shape=[DATA_SIZE], dtype=tf.float64) for _ in range(blowup)]
    return tf.concat(random_arrays, axis=0)


def run_experiment(*, blowup: int = 0, parallelism: int = tf.data.AUTOTUNE, size: int = -1):
    start = time.perf_counter()

    ds = tf.data.Dataset.range(size)
    ds = ds.map(gen_data, num_parallel_calls=parallelism)
    if blowup > 0:
        ds = ds.map(lambda x: memory_blowup(x, blowup), num_parallel_calls=parallelism)
        # ds = ds.flat_map(memory_blowup_flat, fn_kwargs={"blowup": blowup})

    ret = ds.reduce(np.int64(0), lambda total, current: total + tf.size(current, out_type=tf.int64) * tf.float64.size)

    end = time.perf_counter()
    print(f"\n{ret:,}")
    print(f"Time: {end - start:.4f}s")
    return ret
   

def main():
    # run_experiment(parallelism=-1, size=10000, blowup=20)
    # run_experiment(parallelism=10, size=100, blowup=20)
    run_experiment(size=100, blowup=20)


if __name__ == "__main__":
    main()
