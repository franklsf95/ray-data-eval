import csv
import glob
import io
import logging
import time

import numpy as np
from PIL import Image
from pyflink.common.typeinfo import Types
from pyflink.datastream import KeyedProcessFunction, MapFunction, StreamExecutionEnvironment
from pyflink.common import Configuration
import torch
import torchvision.transforms.functional as F
from torchvision.models import ResNet152_Weights
from torchvision import models

EXECUTION_MODE = "process"
LOADER_PARALLELISM = 1
MODEL_PARALLELISM = 1

IMAGENET_LOCAL_DIR = "/mnt/data/ray-data-eval/ILSVRC/Data/CLS-LOC/10k/"
BATCH_SIZE = 256

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def transform_image(image: Image) -> torch.Tensor:
    image = image.resize((232, 232), resample=Image.BILINEAR)
    image = image.convert("RGB")
    image = F.pil_to_tensor(image)
    image = F.center_crop(image, 224)
    image = F.convert_image_dtype(image, torch.float)
    image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image


class Loader(MapFunction):
    def map(self, file_path: str) -> torch.Tensor:
        with open(file_path, "rb") as f:
            data = f.read()
        image = Image.open(io.BytesIO(data))
        image = transform_image(image)
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

        self.csv_logger = CsvLogger("inference.csv")

    def __call__(self, batch: torch.Tensor):
        logging.warning("Inference")
        inference_start_time = time.time()
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(self.device)
        with torch.inference_mode():
            prediction = self.model(batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [self.categories[i] for i in predicted_classes]

        inference_end_time = time.time()
        num_rows = len(batch)
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
        return predicted_labels


class DummyMapper(MapFunction):
    def map(self, value):
        return str(type(value))


class ModelMapper(MapFunction):
    model = None

    def open(self, _runtime_context):
        self.model = ResnetModel()

    def map(self, value):
        batch = value.unsqueeze(0)
        return self.model(batch)


class DummyProcessor(KeyedProcessFunction):
    current_batch = []

    def process_element(self, value, _ctx):
        self.current_batch.append(value)
        if len(self.current_batch) < BATCH_SIZE:
            return
        logging.warning(f"Processing batch of size {len(self.current_batch)}")
        self.current_batch = []
        return ["test"] * BATCH_SIZE


class ModelProcessor(KeyedProcessFunction):
    current_batch = []
    model = None

    def open(self, _runtime_context):
        self.model = ResnetModel()

    def process_element(self, value, _ctx):
        self.current_batch.append(value)
        if len(self.current_batch) < BATCH_SIZE:
            return
        logging.warning(f"Processing batch of size {len(self.current_batch)}")
        batch = torch.stack(self.current_batch, dim=0)
        result = self.model(batch)
        self.current_batch = []
        return result


def get_image_file_paths(root_path: str) -> list[str]:
    ret = []
    extensions = ["jp*g", "png", "gif"]
    extensions.extend([ext.upper() for ext in extensions])
    for ext in extensions:
        ret.extend(glob.iglob(f"{root_path}/**/*.{ext}", recursive=True))
    return ret


def test_without_flink(file_paths: list[str]):
    loader = Loader()
    mapper = ModelMapper()
    mapper.open(None)
    file_path = file_paths[0]
    image = loader.map(file_path)
    result = mapper.map(image)
    print(result)


def run_flink(env):
    file_paths = get_image_file_paths(IMAGENET_LOCAL_DIR)
    print(len(file_paths))

    # test_without_flink(file_paths)

    ds = env.from_collection(file_paths, type_info=Types.STRING())
    loader = Loader()
    ds = (
        ds.map(loader, output_type=Types.PICKLED_BYTE_ARRAY())
        .set_parallelism(LOADER_PARALLELISM)
        .disable_chaining()
    )
    # ds = ds.map(DummyMapper()).set_parallelism(MODEL_PARALLELISM)
    # ds = ds.map(ModelMapper()).set_parallelism(MODEL_PARALLELISM)
    # ds = ds.key_by(lambda x: 0).process(DummyProcessor()).set_parallelism(MODEL_PARALLELISM)
    ds = ds.key_by(lambda x: 0).process(ModelProcessor()).set_parallelism(MODEL_PARALLELISM)
    ds.print()
    env.execute("Image Batch Inference")


def main():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    env = StreamExecutionEnvironment.get_execution_environment(config)

    run_flink(env)


if __name__ == "__main__":
    main()
