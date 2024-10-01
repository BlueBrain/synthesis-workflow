"""Tests for workflows module."""

import shutil

import luigi
import numpy as np
import pytest
from dir_content_diff import assert_equal_trees

from synthesis_workflow.tasks.workflows import ValidateVacuumSynthesis

from . import export_config
from . import get_config_parser


@pytest.mark.xdist_group("group_vacuum")
def test_ValidateVacuumSynthesis(vacuum_working_directory, data_dir):
    """Test the synthesis workflow in vacuum."""
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
                    "replace_pattern": {(data_dir_pattern, ""): ["path", "morphology_path"]}
                }
            },
            "morphs_df/morphs_df.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {(data_dir_pattern, ""): ["path", "morphology_path"]}
                }
            },
            "synthesis/neurots_input/tmd_distributions.json": {
                "tolerance": 2e-3,
                "absolute_tolerance": 1e-15,
            },
        },
    )


@pytest.mark.xdist_group("group_vacuum")
def test_neuronDB(vacuum_working_directory, data_dir, WorkflowTask_exception_event):
    """Test the synthesis workflow with NeuronDB file not located in the morphology folder."""
    root_dir = vacuum_working_directory[0].parent

    # Create a new NeuronDB file outside the directory that contain the morphologies
    neurondb_path = data_dir / "input_cells" / "neuronDB.xml"
    new_neurondb_path = root_dir / "neuronDB.xml"
    shutil.copyfile(neurondb_path, new_neurondb_path)

    # Update config with new NeuronDB path
    params = get_config_parser(root_dir / "luigi.cfg")
    params["BuildMorphsDF"]["neurondb_path"] = (new_neurondb_path).as_posix()
    export_config(params, root_dir / "luigi.cfg")

    luigi_config = luigi.configuration.get_config()
    luigi_config.read(root_dir / "luigi.cfg")

    # Run the workflow
    assert not luigi.build([ValidateVacuumSynthesis()], local_scheduler=True)

    failed_task, exceptions = WorkflowTask_exception_event
    assert len(failed_task) == 1
    assert failed_task[0].startswith("BuildMorphsDF(")
    assert exceptions == [
        "The following morphologies extracted from the MorphDB file do not exist: "
        "['C270106A', 'C170797A-P2', 'rat_20160908_E3_LH2_cell2', 'sm080625a1-6_idD']"
    ]
