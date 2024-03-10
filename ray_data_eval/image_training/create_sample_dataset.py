"""Create a subset of the ImageNet dataset by randomly sampling; useful for benchmark tests."""
import os
import shutil
import random
import multiprocessing


def copy_directory(src_path, dst_path, copy_dir):
    if copy_dir:
        shutil.copytree(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")
    else:
        shutil.copy2(src_path, dst_path)
    


def copy_random_folders(source_dir, target_dir, percentage, copy_dir=True):
    """
    Copies a percentage of folders from the source directory to the target directory.
    (Updated Mar 4: Add multiprocessing to speed up copying.)

    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if copy_dir:
        subdirs = [
            d
            for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]
    else:
        subdirs = [d for d in os.listdir(source_dir)]

    num_to_copy = int(len(subdirs) * percentage)
    print("Number of dir: ", num_to_copy)

    # returns num_to_copy *unique* elements, so no repetition
    selected_dirs = random.sample(subdirs, num_to_copy)

    # init multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    tasks = [
        (os.path.join(source_dir, subdir), os.path.join(target_dir, subdir), copy_dir)
        for subdir in selected_dirs
    ]

    pool.starmap(copy_directory, tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a X percent sample of the ImageNet dataset."
    )
    parser.add_argument(
        "--percentage",
        default=2,
        type=int,
        help=(
            "Specifies the size of the sample to generate, i.e. X percent of the ImageNet dataset training data. 1 percent ~ 1300 images."
        ),
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        type=bool,
        help=(
            "Whether to delete previously generated sample of the same percentage. Defaults to True."
        ),
    )
    args = parser.parse_args()

    PERCENTAGE = args.percentage
    SOURCE_DIR = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/train/"
    TARGET_DIR = (
        f"/home/ubuntu/image-data-{PERCENTAGE}-percent/ILSVRC/Data/CLS-LOC/train/"
    )

    if args.overwrite and os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
        print(f"Deleted old {TARGET_DIR}")

    copy_random_folders(SOURCE_DIR, TARGET_DIR, percentage=PERCENTAGE / 100)

    # make sure test, val directories contents ePERCENTAGEist so that the benchmark doesn't error, although these will not be used
    source_test = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/test/"
    source_val = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/val/"
    test_dir = f"/home/ubuntu/image-data-{PERCENTAGE}-percent/ILSVRC/Data/CLS-LOC/test/"
    val_dir = f"/home/ubuntu/image-data-{PERCENTAGE}-percent/ILSVRC/Data/CLS-LOC/val/"
    copy_random_folders(source_test, test_dir, 0.01, copy_dir=False)
    copy_random_folders(source_val, val_dir, 0.01, copy_dir=False)
