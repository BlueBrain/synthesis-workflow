"""Tests for workflows module."""
import json
from pkg_resources import get_distribution
from pkg_resources import parse_version

import luigi
import numpy as np
import pytest

from synthesis_workflow.tasks.workflows import ValidateVacuumSynthesis

from .tools import compare_tree


def test_ValidateVacuumSynthesis(vacuum_working_directory):
    """Test the synthesis workflow in vacuum"""
    np.random.seed(0)

    # Run the workflow
    assert luigi.build([ValidateVacuumSynthesis()], local_scheduler=True)

    result_dir, expected_dir = vacuum_working_directory

    # Fix the results because the function exponnorm was slightly changed in scipy 1.6
    scipy_version = get_distribution("scipy").version
    if parse_version(scipy_version) < parse_version("1.6"):
        old_loc_diameter_L1_DAC_basal = 2.5907068122583383
        old_scale_diameter_L1_DAC_basal = 0.5242253067960662
        old_a_diameter_L3_TPC_A_basal = 1.0641245580067158
        old_loc_diameter_L3_TPC_A_basal = 3.062658680633035
        old_scale_diameter_L3_TPC_A_basal = 1.4455813774864028

        new_loc_diameter_L1_DAC_basal = 2.5911480845973887
        new_scale_diameter_L1_DAC_basal = 0.524115953761996
        new_a_diameter_L3_TPC_A_basal = 1.0653081127070516
        new_loc_diameter_L3_TPC_A_basal = 3.062230952957633
        new_scale_diameter_L3_TPC_A_basal = 1.4447473960401938
        with open(result_dir / "synthesis/tns_input/tmd_distributions.json", "r") as f:
            tmd_distributions = json.load(f)
            assert tmd_distributions["mtypes"]["L1_DAC"]["diameter"]["diameter_power_relation"][
                "basal"
            ]["params"]["loc"] == pytest.approx(old_loc_diameter_L1_DAC_basal)
            assert tmd_distributions["mtypes"]["L1_DAC"]["diameter"]["diameter_power_relation"][
                "basal"
            ]["params"]["scale"] == pytest.approx(old_scale_diameter_L1_DAC_basal)
            assert tmd_distributions["mtypes"]["L3_TPC:A"]["diameter"]["diameter_power_relation"][
                "basal"
            ]["params"]["a"] == pytest.approx(old_a_diameter_L3_TPC_A_basal)
            assert tmd_distributions["mtypes"]["L3_TPC:A"]["diameter"]["diameter_power_relation"][
                "basal"
            ]["params"]["loc"] == pytest.approx(old_loc_diameter_L3_TPC_A_basal)
            assert tmd_distributions["mtypes"]["L3_TPC:A"]["diameter"]["diameter_power_relation"][
                "basal"
            ]["params"]["scale"] == pytest.approx(old_scale_diameter_L3_TPC_A_basal)

        with open(result_dir / "synthesis/tns_input/tmd_distributions.json", "w") as f:
            tmd_distributions["mtypes"]["L1_DAC"]["diameter"]["diameter_power_relation"]["basal"][
                "params"
            ]["loc"] = new_loc_diameter_L1_DAC_basal
            tmd_distributions["mtypes"]["L1_DAC"]["diameter"]["diameter_power_relation"]["basal"][
                "params"
            ]["scale"] = new_scale_diameter_L1_DAC_basal
            tmd_distributions["mtypes"]["L3_TPC:A"]["diameter"]["diameter_power_relation"]["basal"][
                "params"
            ]["a"] = new_a_diameter_L3_TPC_A_basal
            tmd_distributions["mtypes"]["L3_TPC:A"]["diameter"]["diameter_power_relation"]["basal"][
                "params"
            ]["loc"] = new_loc_diameter_L3_TPC_A_basal
            tmd_distributions["mtypes"]["L3_TPC:A"]["diameter"]["diameter_power_relation"]["basal"][
                "params"
            ]["scale"] = new_scale_diameter_L3_TPC_A_basal
            json.dump(tmd_distributions, f)

    # Check the results
    compare_tree(
        expected_dir,
        result_dir,
        specific_args={
            "morphs_df/vacuum_synth_morphs_df.csv": {
                "kwargs": {"ignore_columns": ["vacuum_synth_morphologies"]}
            },
            "morphs_df/substituted_morphs_df.csv": {
                "kwargs": {"ignore_columns": ["repaired_morphology_path"]}
            },
            "morphs_df/morphs_df.csv": {"kwargs": {"ignore_columns": ["repaired_morphology_path"]}},
            "synthesis/tns_input/tmd_distributions.json": {
                "kwargs": {"precision": 2}  # weird precision issue in CI
            },
            "validation/score_matrix_reports.pdf": {"kwargs": {"threshold": 64.5}},
            "validation/vacuum_morphologies.pdf": {"kwargs": {"threshold": 70}},
        },
    )
