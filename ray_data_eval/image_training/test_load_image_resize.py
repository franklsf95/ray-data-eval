"""
Modified from https://gist.github.com/stephanie-wang/88500403e701537568383ef2e181768c.
Tested on EC2 m7i.2xlarge

Example Output (in `Output.csv`):
- cv2.resize
ray_data,1667.8480287284228
ray_data,1687.4230603087162
ray_data,1673.1990663557342
ray_data,1689.390632496964
ray_data,1678.464523163436

- PIL fromarray + PIL.resize
ray_data,1004.0476897381396
ray_data,1000.6739399326473
ray_data,1012.8306755717965
ray_data,999.2872670749986
ray_data,1021.3997983515267
"""

import ray
import time
from PIL import Image


DEFAULT_IMAGE_SIZE = 224


def iterate(dataset, label, batch_size, output_file=None):
    start = time.time()
    it = iter(dataset)
    num_rows = 0
    print_at = 1000
    for batch in it:
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]
        else:
            batch = batch["image"]
        num_rows += batch.shape[0]
        if num_rows >= print_at:
            print(f"Read {num_rows} rows")
            print_at = ((num_rows // 1000) + 1) * 1000
    end = time.time()
    print(label, end - start, "epoch", i)

    tput = num_rows / (end - start)
    print(label, "tput", tput, "epoch", i)

    if output_file is None:
        output_file = "output.csv"
    with open(output_file, "a+") as f:
        f.write(f"{label},{tput}\n")


def resize_fn(row, mode='cv2'):
    import cv2

    image = row["image"]
    if mode == 'cv2':
        resized_image = cv2.resize(
            image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
        )
    else:
        image = Image.fromarray(image)
        resized_image = image.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    row["image"] = resized_image
    return row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run single-node batch iteration benchmarks."
    )

    parser.add_argument(
        "--data-root",
        default=None,
        type=str,
        help=(
            "Directory path with raw images. Directory structure should be "
            '"<data_root>/train/<class>/<image file>"'
        ),
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--num-epochs",
        default=1,
        type=int,
        help="Number of epochs to run. The throughput for the last epoch will be kept.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        type=str,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.data_root is not None:
        # ray.data, load images.
        # ray.init(num_cpus=8)
        # ray_dataset = ray.data.read_images(
        #     args.data_root,
        #     mode="RGB",
        #     size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        # )

        # for i in range(args.num_epochs):
        #     iterate(
        #         ray_dataset.iter_batches(
        #             batch_size=args.batch_size, prefetch_batches=1
        #         ),
        #         "ray_data",
        #         args.batch_size,
        #         args.output_file,
        #     )
        # print(ray_dataset.stats())
        # ray.shutdown()

        ray.init(num_cpus=8)
        ray_dataset = ray.data.read_images(
            args.data_root,
            mode="RGB",
        ).map(resize_fn)

        for i in range(args.num_epochs):
            iterate(
                ray_dataset.iter_batches(
                    batch_size=args.batch_size, prefetch_batches=1
                ),
                "ray_data",
                args.batch_size,
                args.output_file,
            )
        print(ray_dataset.stats())
        ray.shutdown()
