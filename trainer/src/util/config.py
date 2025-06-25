import os

import yaml


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def merge_configs(primary_config: dict, secondary_config: dict) -> dict:
    """Merge configurations with precedence: primary config > secondary config."""
    merged_config = {}

    for key, value in secondary_config.items():
        if value is not None:
            merged_config[key] = value

    for key, value in primary_config.items():
        if value is not None:
            merged_config[key] = value

    return merged_config
