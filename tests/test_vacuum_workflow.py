"""Tests for workflows module."""
import numpy as np
import luigi

from synthesis_workflow.tasks.workflows import ValidateVacuumSynthesis

from .tools import compare_tree


def test_ValidateVacuumSynthesis(vacuum_working_directory):
    """Test the synthesis workflow in vacuum"""
    np.random.seed(0)

    # Run the workflow
    assert luigi.build([ValidateVacuumSynthesis()], local_scheduler=True)

    # Check the results
    result_dir, expected_dir = vacuum_working_directory
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
            "morphs_df/morphs_df.csv": {
                "kwargs": {"ignore_columns": ["repaired_morphology_path"]}
            },
            "synthesis/tns_input/tmd_distributions.json": {
                "kwargs": {"precision": 2}  # weird precision issue in CI
            },
        },
    )
