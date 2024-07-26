import contextlib
import functools
import io
import os
import time

import humanize
import numpy as np
import ray
from ray.data.block import DataBatch

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
NUM_INPUT_REPEAT = 15


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


def produce_video_slices(row: DataBatch):
    print(
        "[Producer Start]", time.perf_counter(), flush=True
    )  # ray.timeline() not correctly displaying producer end time
    from decord import VideoReader, DECORDError

    path = row["item"][0]
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

        all_frames = np.empty(
            (NUM_INPUT_REPEAT * total_num_frames, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
        )
        index = 0
        for iteration in range(NUM_INPUT_REPEAT):
            for start in range(0, total_num_frames, PRODUCER_NUM_FRAMES):
                end = min(start + PRODUCER_NUM_FRAMES, total_num_frames)
                with timer(f"decode {end - start} frames"):
                    frames = vr.get_batch(range(start, end)).asnumpy()
                num_frames = frames.shape[0]
                all_frames[index : index + num_frames] = frames
                index += num_frames
                print(
                    f"[Iteration {iteration}] Processed frames {start}-{end} for video, shape {frames.shape}"
                )

        print("[Producer End]", time.perf_counter(), flush=True)
        return {"frames": all_frames[:index]}
    except DECORDError as e:
        print(f"Failed to process video: {e}")


def preprocess_video(row: DataBatch) -> DataBatch:
    print("preprocess video")
    busy_loop_for_seconds(0.5)
    frames = row["frames"]
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(list(frames), return_tensors="np")
    arr = ret.data["pixel_values"][0]
    return {"video": arr}


def collate_video_frames(batch: DataBatch) -> DataBatch:
    return {"video": np.concatenate(batch["video"], axis=0)}


def get_video_paths(limit: int = 1) -> list[str]:
    DATA_DIR = "/home/ubuntu/Kinetics700-2020-test"
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    return all_files[:limit]


@timeit
def warmup():
    ds = ray.data.from_items(get_video_paths(limit=4))
    ds = ds.map_batches(produce_video_slices, batch_size=1)
    ds = ds.map_batches(
        preprocess_video,
        batch_size=MODEL_NUM_FRAMES,
        num_cpus=0.99,
    )
    ds.take_all()
    time.sleep(1)


@timeit
def main_job():
    ds = ray.data.from_items(
        get_video_paths(
            limit=1,
        )
    )
    ds = ds.map_batches(
        produce_video_slices,
        batch_size=1,
        concurrency=1,
    )
    ds = ds.map_batches(
        preprocess_video,
        batch_size=MODEL_NUM_FRAMES,
        num_cpus=0.99,
        concurrency=1,
    )
    ds = ds.map_batches(
        Classifier,
        batch_size=BATCH_SIZE * MODEL_NUM_FRAMES,
        num_gpus=1,
        concurrency=1,
        zero_copy_batch=True,
        max_concurrency=2,
    )
    ds.take_all()
    print(ds.stats())


def main():
    """
    Usage:
    > python long_video_pipeline_no_repartitioning.py > long_video_pipeline_no_repartitioning_log.txt 2>&1
    """

    TIMELINE_FILENAME = "long_video_pipeline_no_repartitioning.json"

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = (
        np.prod(MODEL_INPUT_SHAPE) * np.dtype(np.float32).itemsize * 1.001
    )

    # warmup()
    main_job()

    ray.timeline(TIMELINE_FILENAME)


if __name__ == "__main__":
    main()
