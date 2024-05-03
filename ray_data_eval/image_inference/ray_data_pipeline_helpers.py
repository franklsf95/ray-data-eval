import boto3
import json
import ray
import time


class ChromeTracer:
    """
    A simple custom profiler that records event start and end time, to replace Ray's profiler due to observed issues with profiling gpu workload.
    https://github.com/ray-project/ray/blob/master/python/ray/_private/profiling.py#L84

    """

    def __init__(self, log_file, device_name="NVIDIA_A10G"):
        self.log_file = log_file
        self.device_name = device_name
        self.events = []

    def _add_event(self, name, phase, timestamp, cname="rail_load", extra_data=None):
        event = {
            "name": name,
            "ph": phase,
            "ts": timestamp,
            "pid": ray._private.services.get_node_ip_address(),
            "tid": "gpu:" + "NVIDIA_A10G",
            "cname": cname,
            "args": extra_data or {},
        }
        self.events.append(event)

    def __enter__(self):
        self.start_time = time.time() * 1000000
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time() * 1000000
        self._add_event(self.name, "B", self.start_time, self.extra_data)
        self._add_event(self.name, "E", self.end_time)

    def profile(self, name, extra_data=None):
        self.name = name
        self.extra_data = extra_data
        return self

    def save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.events, f)


def append_gpu_timeline(timeline_filename, gpu_timeline_filename):
    """
    Append GPU events log to the main log.

    """
    try:
        with open(timeline_filename, "r") as file:
            timeline = json.load(file)
            assert isinstance(timeline, list)

        with open(gpu_timeline_filename, "r") as gpu_file:
            gpu_timeline = json.load(gpu_file)
            assert isinstance(gpu_timeline, list)

        timeline += gpu_timeline

        with open(timeline_filename, "w") as file:
            json.dump(timeline, file)
    except Exception as e:
        print(f"Error occurred when appending GPU timeline: {e}")


def get_prefixes(bucket_name, prefix):
    """
    Each bucket_name, prefix combination creates a path that leads to a folder,
    which contains training data of the same label.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    prefixes = []
    for response in response_iterator:
        if "CommonPrefixes" in response:
            prefixes.extend(
                [f"s3://{bucket_name}/" + cp["Prefix"] for cp in response["CommonPrefixes"]]
            )

    return prefixes, len(prefixes)


def download_train_directories(
    bucket_name,
    prefix,
    percentage=10,
    output_file="kinetics-train-10-percent.txt",
):
    directories, count = get_prefixes(bucket_name, prefix)
    num_samples = len(directories) * percentage // 100
    directories = directories[:num_samples]

    with open(output_file, "w") as f:
        f.write(repr(directories))
    print(f"Downloaded {num_samples} directories to {output_file}")
    return directories


if __name__ == "__main__":
    bucket_name = "ray-data-eval-us-west-2"
    prefix = "kinetics/k700-2020/train/"
    print(download_train_directories(bucket_name, prefix)[0])
