import argparse
import csv
import io
import time

import numpy as np
from PIL import Image
import ray
import ray._private.internal_api
import torch
import torchvision.transforms.functional as F
from torchvision.models import ResNet152_Weights
from torchvision import models, transforms

from ray_data_pipeline_helpers import (
    ChromeTracer,
    append_gpu_timeline,
    download_train_directories,
)

DATA_PERCENTAGE = 1
BUCKET_NAME = "ray-data-eval-us-west-2"
PREFIX = "imagenet/ILSVRC/Data/CLS-LOC/train/"
IMAGENET_LOCAL_DIR = "/mnt/data/ray-data-eval/ILSVRC/Data/CLS-LOC/10k/"
IMAGENET_S3_FILELIST = f"imagenet-train-{DATA_PERCENTAGE}-percent.txt"
BATCH_SIZE = 512

transform = transforms.Compose(
    [transforms.ToTensor(), ResNet152_Weights.IMAGENET1K_V2.transforms()]
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source",
    default="local",
    help="local or S3",
)
parser.add_argument(
    "--iter",
    action="store_const",
    const="iter_batches",
    dest="mode",
)
parser.add_argument(
    "--map",
    action="store_const",
    const="map_batches",
    dest="mode",
)
parser.set_defaults(mode="iter_batches")
args = parser.parse_args()

if args.source == "local":
    print("Using local data.")
    INPUT_PATH = IMAGENET_LOCAL_DIR
else:
    print("Using S3 data.")
    try:
        with open(IMAGENET_S3_FILELIST, "r") as f:
            INPUT_PATH = eval(f.read())
            print(len(INPUT_PATH))
    except FileNotFoundError:
        INPUT_PATH = download_train_directories(
            bucket_name=BUCKET_NAME,
            prefix=PREFIX,
            percentage=DATA_PERCENTAGE,
        )

ACCELERATOR = "NVIDIA_A10G"
TIMELINE_FILENAME = (
    f"logs/ray_log/image_inference_{args.source}_{args.mode}_{BATCH_SIZE}_{DATA_PERCENTAGE}pct.json"
)
GPU_TIMELINE_FILENAME = (
    f"logs/gpu/image_inference_{args.source}_{args.mode}_{BATCH_SIZE}_{DATA_PERCENTAGE}pct_gpu.json"
)
CSV_FILENAME = (
    f"logs/csv/image_inference_{args.source}_{args.mode}_{BATCH_SIZE}_{DATA_PERCENTAGE}pct.csv"
)


def preprocess_image(row: dict[str, np.ndarray]):
    image_bytes = row["bytes"]
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((232, 232), resample=Image.BILINEAR)
    # image = image.convert("RGB")
    image = F.pil_to_tensor(image)
    image = F.center_crop(image, 224)
    image = F.convert_image_dtype(image, torch.float)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return {
        "image": transform(row["image"]),
    }


def transform_image(image: Image) -> np.ndarray:
    image = image.resize((232, 232), resample=Image.BILINEAR)
    image = image.convert("RGB")
    image = F.pil_to_tensor(image)
    image = F.center_crop(image, 224)
    image = F.convert_image_dtype(image, torch.float)
    image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image


class CsvLogger:
    def __init__(self, filename: str):
        self.filename = filename
        with open(self.filename, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "time_from_start",
                    "total_rows",
                    "cumulative_throughput",
                    "batch_rows",
                    "batch_inference_time",
                    "batch_inference_throughput",
                    "batch_time",
                    "batch_throughput",
                ]
            )
            writer.writerow([0, 0, 0, 0, 0, 0, 0, 0])

    def write_csv_row(self, row):
        with open(self.filename, mode="a") as file:
            writer = csv.writer(file)
            writer.writerow(row)


class ResnetModel:
    def __init__(self):
        weights = ResNet152_Weights.IMAGENET1K_V2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=weights).to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]

        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.total_num_rows = 0

        self.csv_logger = CsvLogger(CSV_FILENAME)

    def __call__(self, batch: dict[str, np.ndarray]):
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        inference_start_time = time.time()
        torch_batch = torch.from_numpy(batch["image"]).to(self.device)
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [self.categories[i] for i in predicted_classes]

        inference_end_time = time.time()
        num_rows = len(batch["image"])
        self.total_num_rows += num_rows
        self.csv_logger.write_csv_row(
            [
                inference_end_time - self.start_time,
                self.total_num_rows,
                self.total_num_rows / (inference_end_time - self.start_time),
                num_rows,
                inference_end_time - inference_start_time,
                num_rows / (inference_end_time - inference_start_time),
                inference_end_time - self.last_end_time,
                num_rows / (inference_end_time - self.last_end_time),
            ]
        )
        self.last_end_time = inference_end_time

        # print(ray._private.internal_api.memory_summary(stats_only=True))
        return {
            "predicted_label": predicted_labels,
        }


def main():
    ds = ray.data.read_images(
        INPUT_PATH,
        transform=transform_image,
        override_num_blocks=50,
    )

    model = None
    if args.mode == "map_batches":
        ds = ds.map_batches(
            ResnetModel,
            concurrency=1,
            num_gpus=1,
            batch_size=BATCH_SIZE,
            zero_copy_batch=True,
            max_concurrency=2,
        )
    else:  # iter_batches
        model = ResnetModel()

    tracer = ChromeTracer(GPU_TIMELINE_FILENAME, ACCELERATOR)
    if args.mode == "map_batches":
        ds.take_all()
    else:
        last_batch_time = time.time()
        for batch in ds.iter_batches(batch_size=BATCH_SIZE):
            print(f"Time to read batch: {time.time() - last_batch_time:.4f}")
            if args.mode == "iter_batches":
                with tracer.profile("task:gpu_execution"):
                    batch = model(batch)

            print(f"Total batch time: {time.time() - last_batch_time:.4f}")
            last_batch_time = time.time()

    print(ds.stats())

    # Save and combine cpu, gpu timeline view
    tracer.save()
    ray.timeline(TIMELINE_FILENAME)
    append_gpu_timeline(TIMELINE_FILENAME, GPU_TIMELINE_FILENAME)
    print("Timeline log saved to: ", TIMELINE_FILENAME)


if __name__ == "__main__":
    # ray.init(object_store_memory=16e9)
    ray.init(num_cpus=5, object_store_memory=8e9)
    main()
    ray.shutdown()
