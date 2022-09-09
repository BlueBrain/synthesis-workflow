"""Configuration for the pytest test suite."""
# pylint: disable=redefined-outer-name
import json
import os
import shutil
from configparser import ConfigParser
from pathlib import Path
from subprocess import call

import dir_content_diff.pandas
import dir_content_diff_plugins.voxcell
import luigi
import numpy as np
import pytest
from neurocollage.planes import get_layer_annotation

from synthesis_workflow.tasks import config

dir_content_diff.pandas.register()
dir_content_diff_plugins.voxcell.register()


TEST_ROOT = Path(__file__).parent
DATA = TEST_ROOT / "data"


def export_config(params, filepath):
    """Export params to a file."""
    with open(filepath, "w", encoding="utf-8") as configfile:
        params.write(configfile)


@pytest.fixture
def root_dir():
    """The root directory."""
    return TEST_ROOT


@pytest.fixture
def data_dir():
    """The data directory."""
    return DATA


@pytest.fixture
def small_O1(tmp_path):
    """Dump a small O1 atlas in folder path."""
    atlas_dir = tmp_path / "small_O1"
    # fmt: off
    with open(os.devnull, "w", encoding="utf-8") as f:
        call(["brainbuilder", "atlases",
              "-n", "6,5,4,3,2,1",
              "-t", "200,100,100,100,100,200",
              "-d", "100",
              "-o", str(atlas_dir),
              "column",
              "-a", "1000",
              ], stdout=f, stderr=f)
    # fmt: on

    # Add metadata
    shutil.copyfile(DATA / "in_small_O1" / "metadata.json", atlas_dir / "metadata.json")

    # Add dummy cell density files for L1_DAC and L3_TPC:A
    br = get_layer_annotation(
        {
            "atlas": str(atlas_dir),
            "structure": str(DATA / "synthesis_input" / "region_structure.yaml"),
        }
    )["annotation"]

    layer_tags = br.raw.copy()
    br.raw[np.where(layer_tags == 1)] = 1000
    br.raw[np.where(layer_tags != 1)] = 0
    br.save_nrrd((atlas_dir / "[cell_density]L1_DAC.nrrd").as_posix())
    br.raw[np.where(layer_tags == 3)] = 1000
    br.raw[np.where(layer_tags != 3)] = 0
    br.save_nrrd((atlas_dir / "[cell_density]L3_TPC:A.nrrd").as_posix())

    return atlas_dir


@pytest.fixture(scope="function")
def tmp_working_dir(tmp_path):
    """Change working directory before a test and change it back when the test is finished."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


def get_config_parser(cfg_path):
    """Return a config parser filed with values from the given file."""
    params = ConfigParser()
    params.read(cfg_path)
    return params


@pytest.fixture
def small_O1_params():
    """Parameters for the small O1 case."""
    return get_config_parser(DATA / "in_small_O1" / "luigi.cfg")


@pytest.fixture
def vacuum_params():
    """Parameters for the vacuum case."""
    return get_config_parser(DATA / "in_vacuum" / "luigi.cfg")


def set_param_paths(params, tmp_working_dir, atlas_path=None):
    """Set proper paths into parameters."""
    if atlas_path is not None:
        params["CircuitConfig"]["atlas_path"] = atlas_path.as_posix()
    params["BuildMorphsDF"]["neurondb_path"] = (DATA / "input_cells" / "neuronDB.xml").as_posix()
    params["BuildMorphsDF"]["morphology_dirs"] = json.dumps(
        {
            "morphology_path": (DATA / "input_cells").as_posix(),
        }
    )
    params["PathConfig"]["result_path"] = (tmp_working_dir / "out").as_posix()
    params["PathConfig"]["local_synthesis_input_path"] = (
        tmp_working_dir / "synthesis_input"
    ).as_posix()


@pytest.fixture
def vacuum_working_directory(tmp_working_dir, vacuum_params):
    """Create the working directory for the vacuum case."""
    shutil.copytree(DATA / "synthesis_input", tmp_working_dir / "synthesis_input")
    shutil.copyfile(DATA / "logging.conf", tmp_working_dir / "logging.conf")

    # Setup config
    params = vacuum_params
    set_param_paths(params, tmp_working_dir)

    # Export config
    export_config(params, "luigi.cfg")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    # Reset output default paths
    config.reset_default_prefixes()

    yield tmp_working_dir / "out", DATA / "in_vacuum" / "out"

    # Reset luigi config
    luigi_config.clear()


@pytest.fixture
def small_O1_working_directory(tmp_working_dir, small_O1_params, small_O1):
    """Create the working directory for the small O1 case."""
    shutil.copytree(DATA / "synthesis_input", tmp_working_dir / "synthesis_input")
    shutil.copyfile(DATA / "logging.conf", tmp_working_dir / "logging.conf")

    # Setup config
    params = small_O1_params
    set_param_paths(params, tmp_working_dir, small_O1)
    params["BuildAxonMorphologies"]["axon_cells_path"] = (DATA / "input_cells").as_posix()

    # Export config
    export_config(params, "luigi.cfg")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    # Reset output default paths
    config.reset_default_prefixes()

    yield tmp_working_dir / "out", DATA / "in_small_O1" / "out", small_O1

    # Reset luigi config
    luigi_config.clear()
