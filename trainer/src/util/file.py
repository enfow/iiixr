import json
from typing import Union

from schema.result import EvalResult, SingleEpisodeResult


def save_json(data: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def log_result(result: Union[SingleEpisodeResult, EvalResult], file_path: str):
    json_dumped = json.dumps(result.to_dict()) + "\n"
    save_json(json_dumped, file_path)


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)
