"""Tests for workflows module."""
import luigi
import numpy as np
import pytest
from dir_content_diff import assert_equal_trees

from synthesis_workflow.tasks.workflows import ValidateVacuumSynthesis


@pytest.mark.xdist_group("group_vacuum")
def test_ValidateVacuumSynthesis(vacuum_working_directory, data_dir):
    """Test the synthesis workflow in vacuum"""
    np.random.seed(0)

    # Run the workflow
    assert luigi.build([ValidateVacuumSynthesis()], local_scheduler=True)

    result_dir, expected_dir = vacuum_working_directory

    data_dir_pattern = str(data_dir) + "/?"
    result_dir_pattern = str(result_dir) + "/?"

    # Check the results
    assert_equal_trees(
        expected_dir,
        result_dir,
        specific_args={
            "morphs_df/vacuum_synth_morphs_df.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {(result_dir_pattern, ""): ["vacuum_synth_morphologies"]}
                }
            },
            "morphs_df/substituted_morphs_df.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        (data_dir_pattern, ""): ["path", "repaired_morphology_path"]
                    }
                }
            },
            "morphs_df/morphs_df.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        (data_dir_pattern, ""): ["path", "repaired_morphology_path"]
                    }
                }
            },
            "synthesis/neurots_input/tmd_distributions.json": {
                "tolerance": 2e-3,
                "absolute_tolerance": 1e-15,
            },
        },
    )
