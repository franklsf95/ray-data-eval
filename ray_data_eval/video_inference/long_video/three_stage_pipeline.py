import argparse
import functools
import io
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


from ray_data_pipeline_helpers import postprocess, download_train_directories

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source",
    default="local",
    help="local or S3",
)

args = parser.parse_args()

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
IMAGE_SIZE = 224
NUM_FRAMES = 16
MODEL_INPUT_SHAPE = (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
BATCH_SIZE = 32

if args.source == "local":
    print("Using local data.")
    INPUT_PATH = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test"
else:
    print("Using S3 data.")
    try:
        with open("kinetics-train-1-percent.txt", "r") as f:
            INPUT_PATH = eval(f.read())
    except FileNotFoundError:
        INPUT_PATH = download_train_directories(
            bucket_name="ray-data-eval-us-west-2", prefix="kinetics/k700-2020/train/", percentage=1
        )


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


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def print_gpu_memory_usage():
    print(
        f"Total GPU memory: {humanize.naturalsize(torch.cuda.get_device_properties(0).total_memory)}"
    )
    print(f"Reserved GPU memory: {humanize.naturalsize(torch.cuda.memory_reserved(0))}")
    print(f"Allocated GPU memory: {humanize.naturalsize(torch.cuda.memory_allocated(0))}")


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
        batch = collate_video_frames(batch)
        model_input = torch.from_numpy(batch["video"]).to(DEVICE)
        print(f"Input tensor size: {tensor_size(model_input)}")
        model_output = self.model(model_input)
        logits = model_output.logits
        preds = logits.argmax(-1)
        result = [self.model.config.id2label[pred.item()] for pred in preds]
        print_gpu_memory_usage()

        inference_end_time = time.time()
        print(
            "[Completed Batch]",
            inference_end_time,
            len(batch["video"]),
            "[Batch Tput]",
            len(batch["video"]) / (inference_end_time - self.last_batch_time),
            "[Inference Tput]",
            len(batch["video"]) / (inference_end_time - inference_start_time),
        )
        self.last_batch_time = inference_end_time
        return {"result": result}


def generate_video_slices(row: DataBatch) -> list[DataBatch]:
    from decord import VideoReader, DECORDError

    video_bytes = row["bytes"]
    ret = []
    try:
        vr = VideoReader(
            io.BytesIO(video_bytes),
            num_threads=1,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        total_num_frames = len(vr)
        for start in range(0, total_num_frames, NUM_FRAMES):
            if start + NUM_FRAMES > total_num_frames:
                break
            ret.append(
                {
                    "bytes": video_bytes,
                    "start": start,
                }
            )
    except DECORDError as e:
        print(f"Failed to process video: {e}")
    return ret


def preprocess_video(row: DataBatch) -> DataBatch:
    from decord import VideoReader, DECORDError

    video_bytes = row["bytes"]
    start = row.get("start", 0)
    end = start + NUM_FRAMES
    try:
        vr = VideoReader(
            io.BytesIO(video_bytes),
            num_threads=1,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        if end > len(vr):
            raise DECORDError("End frame is greater than total frames")
        frames = vr.get_batch(range(start, end)).asnumpy()
    except DECORDError as e:
        print(f"Failed to process video: {e}")
        return {"video": np.zeros((1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)}

    frames = list(frames)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(frames, return_tensors="np")
    arr = ret.data["pixel_values"]

    return {"video": arr}


def collate_video_frames(batch: DataBatch) -> DataBatch:
    return {"video": np.concatenate(batch["video"], axis=0)}


@timeit
def main():
    INSTANCE = "g5_xlarge"
    TIMELINE_FILENAME = f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}.json"
    OUTPUT_FILENAME = f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}.out"

    start_time = time.time()
    print("[Start Time]", start_time)

    ds = ray.data.read_binary_files(
        INPUT_PATH,
    )
    ds = ds.flat_map(generate_video_slices)
    ds = ds.map(preprocess_video, num_cpus=0.99)

    ds = ds.map_batches(
        Classifier,
        batch_size=BATCH_SIZE,
        num_gpus=1,
        concurrency=1,
        zero_copy_batch=True,
        max_concurrency=2,
    )

    ds.take_all()
    print(ds.stats())

    ray.timeline(TIMELINE_FILENAME)

    postprocess(OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
