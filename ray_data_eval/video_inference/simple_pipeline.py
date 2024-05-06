import functools
import io
import os
import time

import humanize
import numpy as np
import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-huge-finetuned-kinetics"
IMAGE_SIZE = 224
NUM_FRAMES = 16

DATA_PATH = "/mnt/data/ray-data-eval/kinetics"
INPUT_PATH = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test"

ModelInputType = torch.Tensor


def print_gpu_memory_usage():
    print(
        f"Total GPU memory: {humanize.naturalsize(torch.cuda.get_device_properties(0).total_memory)}"
    )
    print(f"Allocated GPU memory: {humanize.naturalsize(torch.cuda.memory_allocated(0))}")
    print(f"Reserved GPU memory: {humanize.naturalsize(torch.cuda.memory_reserved(0))}")


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.2f} seconds")
        return result

    return wrapper


@timeit
def load_file(file_path: str) -> io.BytesIO:
    with open(file_path, "rb") as f:
        return io.BytesIO(f.read())


class Classifier:
    def __init__(self):
        start_time = time.time()
        self.model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).eval().to(DEVICE)
        print(f"Time to initialize model: {time.time() - start_time}")

    @timeit
    @torch.no_grad
    def __call__(self, model_input: ModelInputType) -> list:
        print(model_input.shape)
        model_input = model_input.to(DEVICE)
        model_output = self.model(pixel_values=model_input)
        logits = model_output.logits
        preds = logits.argmax(-1)
        print_gpu_memory_usage()
        return [self.model.config.id2label[pred.item()] for pred in preds]


def read_and_decode_video(file_path: str) -> list[np.ndarray]:
    from decord import VideoReader

    vr = VideoReader(file_path, num_threads=1, width=IMAGE_SIZE, height=IMAGE_SIZE)
    frames = vr.get_batch(range(NUM_FRAMES)).asnumpy()
    frames = list(frames)
    return frames


class SimpleIterator:
    def __init__(self):
        self.file_paths = os.listdir(INPUT_PATH)
        self.file_index = 0
        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)

    def __iter__(self):
        return self

    def _preprocess(self, video: list[np.ndarray]) -> ModelInputType:
        ret = self.processor(video, return_tensors="pt")
        return ret.data["pixel_values"]

    def __next__(self) -> ModelInputType:
        if self.file_index >= len(self.file_paths):
            raise StopIteration
        file_path = os.path.join(INPUT_PATH, self.file_paths[self.file_index])
        video = read_and_decode_video(file_path)
        result = self._preprocess(video)
        self.file_index += 1
        return result


class BatchIterator:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self._iter = SimpleIterator()

    def __iter__(self):
        return self

    @timeit
    def __next__(self) -> ModelInputType:
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(next(self._iter))
            except StopIteration:
                break
        if not batch:
            raise StopIteration
        return torch.cat(batch, dim=0)


def main():
    n_batches = 0
    batch_size = 64
    classifier = Classifier()
    iterator = BatchIterator(batch_size=batch_size)
    for batch in iterator:
        classifier(batch)
        n_batches += 1
        if n_batches >= 3:
            break


if __name__ == "__main__":
    main()
