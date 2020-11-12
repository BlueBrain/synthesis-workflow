"""config tools for MorphVal package"""

import os

import yaml


def load_config(config_path):
    """Load configuration from a YAML file"""
    assert os.path.exists(config_path), "Missing config at: %s" % str(config_path)
    #  TODO: Should perform validation of config
    with open(config_path, "r") as fd:
        return yaml.safe_load(fd)["config"]