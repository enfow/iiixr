import os

import yaml


def load_config_from_yaml(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def merge_configs(primary_config: dict, secondary_config: dict) -> dict:
    merged_config = {}

    for key, value in secondary_config.items():
        if value is not None:
            merged_config[key] = value

    for key, value in primary_config.items():
        if value is not None:
            merged_config[key] = value

    if "model" in merged_config and isinstance(merged_config["model"], dict):
        if "model" in primary_config and isinstance(primary_config["model"], dict):
            if "model" in secondary_config and isinstance(
                secondary_config["model"], dict
            ):
                merged_model_config = {}
                for k, v in secondary_config["model"].items():
                    if v is not None:
                        merged_model_config[k] = v
                for k, v in primary_config["model"].items():
                    if v is not None:
                        merged_model_config[k] = v
                merged_config["model"] = merged_model_config

    return merged_config
