import time
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import psutil


class Logger:
    def __init__(self, filename: str):
        self._filename = filename
        self._start_time = time.time()

    def record_start(self):
        self._start_time = time.time()
        with open(self._filename, "w"):
            pass

    def log(self, payload: dict):
        payload = {
            **payload,
            "time": time.time() - self._start_time,
            "clock_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"),
        }
        with open(self._filename, "a") as f:
            f.write(json.dumps(payload) + "\n")


def get_process_and_children_memory_usage_in_bytes():
    # Get the main process
    main_process = psutil.Process()

    # Initialize total memory usage with the main process's memory usage
    total_memory = main_process.memory_info().rss

    # Loop through all child processes
    for child in main_process.children(recursive=True):  # recursive=True includes all descendants
        try:
            total_memory += child.memory_info().rss
        except psutil.NoSuchProcess:
            # If the child process terminates before accessing its memory
            continue

    return total_memory


def plot_from_jsonl(file_path, output_path):
    # Read the JSONL file into a DataFrame

    data = pd.read_json(file_path, lines=True)

    # Initialize the 'finished_sum' and set the previous sum to 0 (or any suitable starting value)
    current_sum = 0
    finished_sums = []

    # Calculate finished_sum and update the previous_finished_sum
    for index, row in data.iterrows():
        current_sum += (
            1 if pd.notna(row["producer_finished"]) or pd.notna(row["consumer_finished"]) else 0
        )
        finished_sums.append(current_sum)

    data["finished_sum"] = finished_sums

    memory_usage_times = []
    memory_usage_values = []
    for _, row in data.iterrows():
        if pd.notna(row["memory_usage_in_bytes"]):
            memory_usage_times.append(row["time"])
            memory_usage_values.append(row["memory_usage_in_bytes"] / (1024 * 1024))

    # Set up the plot with two y-axes
    fig, ax1 = plt.subplots()

    # Plot finished sum on the left y-axis
    color = "tab:red"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Finished Tasks (Producer + Consumer)", color=color)
    ax1.plot(data["time"], data["finished_sum"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a second y-axis for memory usage
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Memory Usage (MB)", color=color)
    ax2.plot(memory_usage_times, memory_usage_values, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    # Show the plot
    plt.savefig(output_path)
