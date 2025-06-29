import json
from typing import Union

from schema.result import EvalResult, SingleEpisodeResult


def save_json(data: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def append_json_line(data: dict, file_path: str):
    with open(file_path, "a") as f:
        json.dump(data, f)
        f.write("\n")


def log_result(result: Union[SingleEpisodeResult, EvalResult], file_path: str):
    result_dict = result.to_log_dict()
    append_json_line(result_dict, file_path)


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl(file_path: str):
    results = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))
    except FileNotFoundError:
        # File doesn't exist yet, return empty list
        pass
    return results
