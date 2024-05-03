import contextlib
import functools
import io
import os
import time

import humanize
import numpy as np
import ray
from ray.data.block import DataBatch

import tensorflow as tf
import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
IMAGE_SIZE = 224
MODEL_NUM_FRAMES = 16
MODEL_INPUT_SHAPE = (MODEL_NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
PRODUCER_NUM_FRAMES = MODEL_NUM_FRAMES * 8
PRODUCER_OUTPUT_SHAPE = (PRODUCER_NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 32
NUM_INPUT_REPEAT = 1
TF_PROFILER_LOGS = "logs/tf"


def timeit(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"{name or func.__name__}: {time.time() - start:.2f} seconds")
            return result

        return wrapper

    if callable(name):
        return decorator(name)
    else:
        return decorator


@contextlib.contextmanager
def timer(description: str = ""):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{description} took {end - start:.2f} seconds")


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def print_gpu_memory_usage():
    print(
        f"Total GPU memory: {humanize.naturalsize(torch.cuda.get_device_properties(0).total_memory)}"
    )
    print(f"Reserved GPU memory: {humanize.naturalsize(torch.cuda.memory_reserved(0))}")
    print(f"Allocated GPU memory: {humanize.naturalsize(torch.cuda.memory_allocated(0))}")


def busy_loop_for_seconds(time_diff):
    start = time.perf_counter()
    i = 0
    while time.perf_counter() - start < time_diff:
        i += 1
        continue


class Classifier:
    def __init__(self):
        start_time = time.time()
        self.model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).eval().to(DEVICE)
        print(f"Time to initialize model: {time.time() - start_time}")
        self.last_batch_time = start_time

    @timeit("Inference")
    @torch.no_grad
    def __call__(self, batch: DataBatch) -> DataBatch:
        inference_start_time = time.time()
        batch = batch["video"]
        batch = batch[: (batch.shape[0] // 16) * 16]  # align to 16 frames
        batch = batch.reshape(-1, 16, 3, 224, 224)
        model_input = torch.from_numpy(batch).to(DEVICE)
        print(f"Input tensor size: {tensor_size(model_input)}, shape {model_input.shape}")
        model_output = self.model(model_input)
        logits = model_output.logits
        preds = logits.argmax(-1)
        result = [self.model.config.id2label[pred.item()] for pred in preds]
        print_gpu_memory_usage()

        inference_end_time = time.time()
        print(
            "[Completed Batch]",
            inference_end_time,
            len(batch),
            "[Batch Tput]",
            len(batch) / (inference_end_time - self.last_batch_time),
            "[Inference Tput]",
            len(batch) / (inference_end_time - inference_start_time),
        )
        self.last_batch_time = inference_end_time
        print(result)
        return {"result": result}


def produce_video_slices(path: str):
    from decord import VideoReader, DECORDError

    with open(path, "rb") as fin:
        video_bytes = fin.read()

    try:
        vr = VideoReader(
            io.BytesIO(video_bytes),
            num_threads=1,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        total_num_frames = len(vr)
        print(f"Total number of frames: {total_num_frames}")
        for iteration in range(NUM_INPUT_REPEAT):
            for start in range(0, total_num_frames, PRODUCER_NUM_FRAMES):
                if start + PRODUCER_NUM_FRAMES > total_num_frames:
                    break
                with timer(f"decode {PRODUCER_NUM_FRAMES} frames"):
                    frames = vr.get_batch(range(start, start + PRODUCER_NUM_FRAMES)).asnumpy()
                print(
                    f"[Iteration {iteration}] Yielded frames {start}-{start + PRODUCER_NUM_FRAMES} for video, shape {frames.shape}"
                )
                yield {"frames": frames}
    except DECORDError as e:
        print(f"Failed to process video: {e}")


def preprocess_video(frames) -> DataBatch:
    busy_loop_for_seconds(0.5)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(list(frames), return_tensors="np")
    arr = ret.data["pixel_values"][0]
    return arr


def collate_video_frames(batch: DataBatch) -> DataBatch:
    return {"video": np.concatenate(batch["video"], axis=0)}


def get_video_paths(limit: int = 1) -> list[str]:
    DATA_DIR = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test"
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    return all_files[:limit]


def configure_tf():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


@timeit
def main():
    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = (
        np.prod(MODEL_INPUT_SHAPE) * np.dtype(np.float32).itemsize * 1.001
    )

    options = tf.data.Options()
    options.autotune.enabled = True
    options.autotune.cpu_budget = 8

    configure_tf()

    items = get_video_paths(
        limit=20,
    )
    ds = tf.data.Dataset.from_tensor_slices(items)
    ds = ds.with_options(options).interleave(
        lambda item: tf.data.Dataset.from_generator(
            produce_video_slices,
            args=(item,),
            output_signature={
                "frames": tf.TensorSpec(shape=PRODUCER_OUTPUT_SHAPE, dtype=tf.float32),
            },
            name="producer",
        ),
        cycle_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="producer_flat_map",
    )
    ds = ds.with_options(options).map(
        lambda item: tf.numpy_function(
            preprocess_video,
            inp=[item["frames"]],
            Tout=tf.float32,
            name="consumer",
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="consumer_map",
    )
    start = time.perf_counter()
    classifier = Classifier()
    for row in ds:
        print(time.perf_counter() - start)
        arr = row.numpy()
        classifier({"video": arr})


if __name__ == "__main__":
    if not os.path.exists(TF_PROFILER_LOGS):
        os.makedirs(TF_PROFILER_LOGS)
    tf.profiler.experimental.start(TF_PROFILER_LOGS)
    main()
    tf.profiler.experimental.stop()
