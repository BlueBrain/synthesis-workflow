"""Tests for utils module."""

import json

import pandas as pd

from synthesis_workflow.utils import apply_parameter_diff
from synthesis_workflow.utils import create_parameter_diff


def test_create_parameter_diff(data_dir):
    """Test we create parameter diff."""
    with open(data_dir / "raw_tmd_parameters.json", encoding="utf-8") as f:
        raw_params = json.load(f)
    with open(data_dir / "complete_tmd_parameters.json", encoding="utf-8") as f:
        complete_params = json.load(f)

    custom_parameters = create_parameter_diff(raw_params, complete_params)
    expected_custom_parameters = pd.read_csv(data_dir / "custom_parameters.csv")
    for gid in custom_parameters.index:
        assert custom_parameters.loc[gid, "entry"] == expected_custom_parameters.loc[gid, "entry"]
        assert (
            str(custom_parameters.loc[gid, "value"]) == expected_custom_parameters.loc[gid, "value"]
        )


def test_apply_parameter_diff(data_dir):
    """Test we apply parameter diff and have no diff parameters."""
    with open(data_dir / "raw_tmd_parameters.json", encoding="utf-8") as f:
        raw_params = json.load(f)
    with open(data_dir / "complete_tmd_parameters.json", encoding="utf-8") as f:
        complete_params = json.load(f)

    custom_parameters = pd.read_csv(data_dir / "custom_parameters.csv")
    apply_parameter_diff(raw_params, custom_parameters)
    assert len(create_parameter_diff(raw_params, complete_params).index) == 0
