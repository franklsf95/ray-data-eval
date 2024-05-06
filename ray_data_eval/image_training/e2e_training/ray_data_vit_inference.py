import ray
import numpy as np
import datetime
import time
from PIL import Image
from typing import Dict
from transformers import pipeline

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/val/"
BATCH_SIZE = 64
FULL_DATASET_SIZE = 3925


def save_ray_timeline():
    timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"/tmp/ray-timeline-{timestr}.json"
    ray.timeline(filename=filename)
    print(f"Execution trace saved to {filename}")


class ImageClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "image-classification", model="google/vit-base-patch16-224", device="cuda:0"
        )

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a list of PIL images which is the format the HF pipeline expects.
        outputs = self.classifier(
            [Image.fromarray(image_array) for image_array in batch["image"]],
            top_k=1,
            batch_size=BATCH_SIZE,
        )

        # `outputs` is a list of length-one lists. For example:
        # [[{'score': '...', 'label': '...'}], ..., [{'score': '...', 'label': '...'}]]
        batch["score"] = [output[0]["score"] for output in outputs]
        batch["label"] = [output[0]["label"] for output in outputs]
        return batch


if __name__ == "__main__":
    ray.init(num_cpus=4, object_store_memory=int(8e9))

    start_time = time.time()
    # load imagenette from s3
    # ds = ray.data.read_images(s3_uri, mode="RGB")

    # load imagenette from local
    local_dir = "/home/ubuntu/imagenette2/val/"
    ds = ray.data.read_images(local_dir, mode="RGB")  # override_num_blocks=

    # inference with Vision Transformer
    predictions = ds.map_batches(
        ImageClassifier,
        concurrency=1,  # Use 1 GPU (number of GPUs in your cluster)
        num_gpus=1,  # number of GPUs needed for each ImageClassifier instance
        batch_size=BATCH_SIZE,  # Use the largest batch size that can fit on our GPUs
    )

    num_images = 0
    inf_start_time = time.time()

    for batch in predictions.iter_batches(batch_size=BATCH_SIZE):
        batch = batch["image"]
        num_images += batch.shape[0]

    end_time = time.time()
    inf_total_time = end_time - inf_start_time
    total_time = end_time - start_time
    tput = num_images / inf_total_time
    print("Inference time: ", inf_total_time)
    print("Total images processed: ", num_images)
    print("Inference tput: ", tput)

    print("Total time: ", total_time)

    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))  # check whether spilled

    save_ray_timeline()

    output_file = "output.csv"
    with open(output_file, "a+") as f:
        f.write(f"Batch size: {BATCH_SIZE}, tput: {tput}, inference time {inf_total_time}\n")

    ray.shutdown()
