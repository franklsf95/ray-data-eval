import time

MB = 1024 * 1024
GB = 1024 * MB
TIME_UNIT = 0.5
NUM_CPUS = 8
NUM_GPUS = 4
FRAMES_PER_VIDEO = 5
NUM_VIDEOS = 160
NUM_FRAMES_TOTAL = FRAMES_PER_VIDEO * NUM_VIDEOS
FRAME_SIZE_B = 100 * MB
EXECUTION_MODE = "process"


def busy_loop(time_in_s):
    end_time = time.time() + time_in_s
    while time.time() < end_time:
        pass
