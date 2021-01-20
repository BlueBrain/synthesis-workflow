"""Tests for workflows module."""
import numpy as np
import luigi

from synthesis_workflow.tasks.workflows import ValidateSynthesis

from .tools import compare_tree


def test_ValidateSynthesis(small_O1_working_directory):
    """Test the synthesis workflow in simple atlas"""
    np.random.seed(0)
    assert luigi.build([ValidateSynthesis()], local_scheduler=True)

    # Check the results
    result_dir, expected_dir, small_O1 = small_O1_working_directory
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
