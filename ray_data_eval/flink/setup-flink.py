import os
import subprocess
import yaml

NUM_TASK_MANAGERS = os.cpu_count() + 1
FLINK_PATH = "/opt/flink/"


def update_workers_file(num_task_managers: int = 1):
    with open(FLINK_PATH + "conf/workers", "w") as file:
        file.writelines(["localhost\n"] * num_task_managers)


def read_flink_config():
    with open(FLINK_PATH + "conf/config.yaml") as file:
        return yaml.safe_load(file)


def start_flink(num_task_managers: int = 1):
    # First shut down the existing cluster and taskmanagers
    print(" [Shutting down all existing taskmanagers.]")
    subprocess.run([FLINK_PATH + "bin/taskmanager.sh", "stop-all"], check=True)
    subprocess.run([FLINK_PATH + "bin/stop-cluster.sh"], check=True)
    subprocess.run([FLINK_PATH + "bin/historyserver.sh", "stop"], check=True)

    # Initialize the standalone cluster
    # By modifying the workers file, we initialize the correct number of taskmanagers
    print(" [Starting a standalone Flink cluster.]")
    update_workers_file(num_task_managers)

    subprocess.run([FLINK_PATH + "bin/start-cluster.sh"], check=True)
    subprocess.run([FLINK_PATH + "bin/historyserver.sh", "start"], check=True)


def main():
    start_flink(NUM_TASK_MANAGERS)

    conf = read_flink_config()
    num_task_slots = conf["taskmanager"]["numberOfTaskSlots"]
    print(f"Initialized {NUM_TASK_MANAGERS} TaskManagers.")
    print(f"Each TaskManager has {num_task_slots} task slots.")
    print(f"Total number of task slots: {NUM_TASK_MANAGERS * num_task_slots}")


if __name__ == "__main__":
    main()
