"""
Modified from https://gist.github.com/stephanie-wang/88500403e701537568383ef2e181768c.
Tested on EC2 m7i.2xlarge

Example Output (in `Output.csv`):
- Default (BICUBIC)
ray_data,1137.3931758979813
ray_data,1146.6874856299994
ray_data,1147.080762189496
ray_data,1153.9882179831754
ray_data,1140.6351740671025

- BILINEAR
ray_data,1399.06477460088
ray_data,1381.023590432492
ray_data,1394.170545465624
ray_data,1396.9916050718184
ray_data,1389.6461855214884
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
        ray.init(num_cpus=8)
        ray_dataset = ray.data.read_images(
            args.data_root,
            mode="RGB",
            size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        )

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
