import requests
import json

FLINK_DASHBOARD = "http://localhost:8082"


def get_job_id(base_url=FLINK_DASHBOARD):
    url = f"{base_url}/jobs"
    response = requests.get(url)
    data = response.json()
    return data["jobs"][0]["id"]


JOB_ID = get_job_id()


def get_taskmanager_ids(base_url=FLINK_DASHBOARD):
    url = f"{base_url}/taskmanagers"
    response = requests.get(url)
    data = response.json()
    return [tm["id"] for tm in data["taskmanagers"]]


def get_taskmanager_stdout(tm_id, base_url=FLINK_DASHBOARD):
    url = f"{base_url}/taskmanagers/{tm_id}/stdout"
    response = requests.get(url)
    return response.text


def parse_stdout_to_json(stdout, tm_id):
    log_entries = stdout.split("\n")
    parsed = []
    for entry in log_entries[:-1]:
        json_obj = json.loads(entry[len("WARNING:root:") :])

        json_obj["pid"] = JOB_ID
        json_obj["tid"] = "taskmanager:" + tm_id
        parsed.append(json_obj)

    return parsed


def main():
    out = []
    taskmanager_ids = get_taskmanager_ids()

    for tm_id in taskmanager_ids:
        stdout = get_taskmanager_stdout(tm_id)
        parsed_logs = parse_stdout_to_json(stdout, tm_id)
        out.extend(parsed_logs)

    with open("data.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
