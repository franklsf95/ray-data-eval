import functools
import time

import decord
import humanize
from transformers import VideoMAEImageProcessor

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
# INPUT_FILE = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test/-LO2DhhIdp0_000111_000121.mp4"
INPUT_FILE = "/mnt/data/ray-data-eval/kinetics/1bUab7pXlRg_000299_000309.mp4"

IMAGE_SIZE = 1080


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


@timeit
def load_video(num_frames: int):
    with open(INPUT_FILE, "rb") as fin:
        vr = decord.VideoReader(
            fin,
            num_threads=8,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        print(f"Total frames: {len(vr)}")
        return vr.get_batch(range(0, num_frames)).asnumpy()


@timeit
def preprocess(frames, repeat: int = 1):
    frames = list(frames)
    frames *= repeat
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(frames, return_tensors="np")
    arr = ret.data["pixel_values"]
    print(humanize.naturalsize(arr.nbytes))
    return arr


for num_frames in [1, 16, 64]:
    print(f"Decode {num_frames} frames")
    frames = load_video(num_frames)
    print(f"Preprocess {num_frames} frames")
    preprocess(frames)
    print(f"Preprocess {num_frames} frames x 16")
    preprocess(frames, repeat=16)
