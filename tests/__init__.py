"""Tests suite for the synthesis-workflow package."""

from configparser import ConfigParser


def get_config_parser(cfg_path):
    """Return a config parser filed with values from the given file."""
    params = ConfigParser()
    params.read(cfg_path)
    return params


def export_config(params, filepath):
    """Export params to a file."""
    with open(filepath, "w", encoding="utf-8") as configfile:
        params.write(configfile)
