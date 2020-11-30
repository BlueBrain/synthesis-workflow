"""Tests for workflows module."""
import luigi
import numpy as np
from dir_content_diff import assert_equal_trees

from synthesis_workflow.tasks.workflows import ValidateSynthesis


def test_ValidateSynthesis(small_O1_working_directory, data_dir):
    """Test the synthesis workflow in simple atlas"""
    np.random.seed(0)

    # Run the workflow
    # assert luigi.build([ValidateSynthesis()], local_scheduler=True)

    result_dir, expected_dir, small_O1 = small_O1_working_directory

    data_dir_pattern = str(data_dir) + "/?"
    result_dir_pattern = str(result_dir) + "/?"

    # Check the results
    """
    assert_equal_trees(
        expected_dir,
        result_dir,
        specific_args={
            "morphs_df/synth_morphs_df.csv": {
                "kwargs": {"replace_pattern": {(result_dir_pattern, ""): ["synth_morphology_path"]}}
            },
            "morphs_df/substituted_morphs_df.csv": {
                "kwargs": {
                    "replace_pattern": {
                        (data_dir_pattern, ""): ["path", "repaired_morphology_path"]
                    }
                }
            },
            "morphs_df/morphs_df.csv": {
                "kwargs": {
                    "replace_pattern": {
                        (data_dir_pattern, ""): ["path", "repaired_morphology_path"]
                    }
                }
            },
            "morphs_df/axon_morphs_df.csv": {
                "kwargs": {"replace_pattern": {(data_dir_pattern, ""): ["path", "clone_path"]}}
            },
            "synthesis/apical_points.yaml": {"kwargs": {"tolerance": 1e-9}},
            "synthesis/tns_input/tmd_distributions.json": {"kwargs": {"tolerance": 0.85}},
            "validation/morphology_validation_reports/validation_results.json": {
                "kwargs": {"tolerance": 1e-3}
            },
        },
    )
    """
