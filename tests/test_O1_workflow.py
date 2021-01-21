"""Tests for workflows module."""
import json
from pkg_resources import get_distribution
from pkg_resources import parse_version

import luigi
import numpy as np
import pytest

from synthesis_workflow.tasks.workflows import ValidateSynthesis

from .tools import compare_tree


def test_ValidateSynthesis(small_O1_working_directory):
    """Test the synthesis workflow in simple atlas"""
    np.random.seed(0)

    # Run the workflow
    assert luigi.build([ValidateSynthesis()], local_scheduler=True)

    result_dir, expected_dir, small_O1 = small_O1_working_directory

    # Fix the results because the function exponnorm was slightly changed in scipy 1.6
    scipy_version = get_distribution("scipy").version
    if parse_version(scipy_version) < parse_version("1.6"):
        old_loc_diameter_L1_DAC_basal = 2.5907068122583383
        old_scale_diameter_L1_DAC_basal = 0.5242253067960662
        old_loc_tapers_L1_DAC_basal = -9.251302943365424e-18
        old_max_tapers_L1_DAC_basal = 9.251302943365424e-18
        old_a_diameter_L3_TPC_A_basal = 1.0641245580067158
        old_loc_diameter_L3_TPC_A_basal = 3.062658680633035
        old_scale_diameter_L3_TPC_A_basal = 1.4455813774864028
        old_loc_tapers_L3_TPC_A_apical = -2.4604883719426155e-17
        old_max_tapers_L3_TPC_A_apical = 2.4604883719426155e-17
        old_loc_tapers_L3_TPC_A_basal = -6.818714755199015e-17
        old_max_tapers_L3_TPC_A_basal = 6.818714755199015e-17

        with open(expected_dir / "synthesis/tns_input/tmd_distributions.json", "r") as f:
            expected = json.load(f)

            expected_diam_L1 = expected["mtypes"]["L1_DAC"]["diameter"]
            expected_diam_L3 = expected["mtypes"]["L3_TPC:A"]["diameter"]

            expected_L1_dpr = expected_diam_L1["diameter_power_relation"]
            expected_L1_tap = expected_diam_L1["tapers"]
            expected_L3_dpr = expected_diam_L3["diameter_power_relation"]
            expected_L3_tap = expected_diam_L3["tapers"]

            new_loc_diameter_L1_DAC_basal = expected_L1_dpr["basal"]["params"]["loc"]
            new_scale_diameter_L1_DAC_basal = expected_L1_dpr["basal"]["params"]["scale"]
            new_loc_tapers_L1_DAC_basal = expected_L1_tap["basal"]["params"]["loc"]
            new_max_tapers_L1_DAC_basal = expected_L1_tap["basal"]["params"]["max"]
            new_a_diameter_L3_TPC_A_basal = expected_L3_dpr["basal"]["params"]["a"]
            new_loc_diameter_L3_TPC_A_basal = expected_L3_dpr["basal"]["params"]["loc"]
            new_scale_diameter_L3_TPC_A_basal = expected_L3_dpr["basal"]["params"]["scale"]
            new_loc_tapers_L3_TPC_A_apical = expected_L3_tap["apical"]["params"]["loc"]
            new_max_tapers_L3_TPC_A_apical = expected_L3_tap["apical"]["params"]["max"]
            new_loc_tapers_L3_TPC_A_basal = expected_L3_tap["basal"]["params"]["loc"]
            new_max_tapers_L3_TPC_A_basal = expected_L3_tap["basal"]["params"]["max"]

        with open(result_dir / "synthesis/tns_input/tmd_distributions.json", "r") as f:
            result = json.load(f)

            diam_L1 = result["mtypes"]["L1_DAC"]["diameter"]
            diam_L3 = result["mtypes"]["L3_TPC:A"]["diameter"]

            L1_dpr = diam_L1["diameter_power_relation"]
            L1_tap = diam_L1["tapers"]
            L3_dpr = diam_L3["diameter_power_relation"]
            L3_tap = diam_L3["tapers"]

            assert diam_L1["diameter_power_relation"]["basal"]["params"]["loc"] == pytest.approx(
                old_loc_diameter_L1_DAC_basal
            )
            assert diam_L1["diameter_power_relation"]["basal"]["params"]["scale"] == pytest.approx(
                old_scale_diameter_L1_DAC_basal
            )
            assert diam_L1["tapers"]["basal"]["params"]["loc"] == pytest.approx(
                old_loc_tapers_L1_DAC_basal
            )
            assert diam_L1["tapers"]["basal"]["params"]["max"] == pytest.approx(
                old_max_tapers_L1_DAC_basal
            )
            assert diam_L3["diameter_power_relation"]["basal"]["params"]["a"] == pytest.approx(
                old_a_diameter_L3_TPC_A_basal
            )
            assert diam_L3["diameter_power_relation"]["basal"]["params"]["loc"] == pytest.approx(
                old_loc_diameter_L3_TPC_A_basal
            )
            assert diam_L3["diameter_power_relation"]["basal"]["params"]["scale"] == pytest.approx(
                old_scale_diameter_L3_TPC_A_basal
            )
            assert diam_L3["tapers"]["apical"]["params"]["loc"] == pytest.approx(
                old_loc_tapers_L3_TPC_A_apical
            )
            assert diam_L3["tapers"]["apical"]["params"]["max"] == pytest.approx(
                old_max_tapers_L3_TPC_A_apical
            )
            assert diam_L3["tapers"]["basal"]["params"]["loc"] == pytest.approx(
                old_loc_tapers_L3_TPC_A_basal
            )
            assert diam_L3["tapers"]["basal"]["params"]["max"] == pytest.approx(
                old_max_tapers_L3_TPC_A_basal
            )

        with open(result_dir / "synthesis/tns_input/tmd_distributions.json", "w") as f:
            diam_L1["diameter_power_relation"]["basal"]["params"][
                "loc"
            ] = new_loc_diameter_L1_DAC_basal
            diam_L1["diameter_power_relation"]["basal"]["params"][
                "scale"
            ] = new_scale_diameter_L1_DAC_basal
            diam_L1["tapers"]["basal"]["params"]["loc"] = new_loc_tapers_L1_DAC_basal
            diam_L1["tapers"]["basal"]["params"]["max"] = new_max_tapers_L1_DAC_basal
            diam_L3["diameter_power_relation"]["basal"]["params"][
                "a"
            ] = new_a_diameter_L3_TPC_A_basal
            diam_L3["diameter_power_relation"]["basal"]["params"][
                "loc"
            ] = new_loc_diameter_L3_TPC_A_basal
            diam_L3["diameter_power_relation"]["basal"]["params"][
                "scale"
            ] = new_scale_diameter_L3_TPC_A_basal
            diam_L3["tapers"]["apical"]["params"]["loc"] = new_loc_tapers_L3_TPC_A_apical
            diam_L3["tapers"]["apical"]["params"]["max"] = new_max_tapers_L3_TPC_A_apical
            diam_L3["tapers"]["basal"]["params"]["loc"] = new_loc_tapers_L3_TPC_A_basal
            diam_L3["tapers"]["basal"]["params"]["max"] = new_max_tapers_L3_TPC_A_basal

            json.dump(result, f)

    # Check the results
    compare_tree(
        expected_dir,
        result_dir,
        specific_args={
            "morphs_df/synth_morphs_df.csv": {
                "kwargs": {"ignore_columns": ["synth_morphology_path"]}
            },
            "morphs_df/substituted_morphs_df.csv": {
                "kwargs": {"ignore_columns": ["repaired_morphology_path"]}
            },
            "morphs_df/morphs_df.csv": {"kwargs": {"ignore_columns": ["repaired_morphology_path"]}},
            "morphs_df/axon_morphs_df.csv": {"kwargs": {"ignore_columns": ["clone_path"]}},
            "synthesis/tns_input/tmd_distributions.json": {
                "kwargs": {"precision": 2}  # weird precision issue in CI
            },
            "validation/morphology_validation_reports/validation_results.json": {
                "kwargs": {"precision": 1}  # weird precision issue in CI
            },
        },
    )
