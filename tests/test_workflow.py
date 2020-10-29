"""Tests for workflows module."""
import filecmp
import json
import numpy as np
import luigi
import pandas as pd
import yaml
from diff_pdf_visually import pdfdiff

from synthesis_workflow.tasks.workflows import ValidateSynthesis
from synthesis_workflow.tasks.workflows import ValidateVacuumSynthesis

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
            "morphs_df/morphs_df.csv": {
                "kwargs": {"ignore_columns": ["repaired_morphology_path"]}
            },
            "morphs_df/axon_morphs_df.csv": {
                "kwargs": {"ignore_columns": ["clone_path"]}
            },
            "synthesis/tns_input/tmd_distributions.json": {
                "kwargs": {"precision": 2}  # weird precision issue in CI
            },
            "validation/morphology_validation_reports/validation_results.json": {
                "kwargs": {"precision": 1}  # weird precision issue in CI
            },
        },
    )


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
