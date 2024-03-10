import os

def count_items_in_directory(directory):
    try:
        # num_items = len(os.listdir(directory))
        num_items = 0
        for item in os.listdir(directory):
            sub = os.path.join(directory, item)
            num_items += len(os.listdir(sub))
        return num_items
    except FileNotFoundError:
        return "Directory does not exist."

directory_path = "/home/ubuntu/image-data-2-percent/ILSVRC/Data/CLS-LOC/train/"

# Get the number of items in the directory
num_items = count_items_in_directory(directory_path)

print(f"Number of items in {directory_path}: {num_items}")
