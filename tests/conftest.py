import json
import os
import shutil
from configparser import ConfigParser
from pathlib import Path
from subprocess import call

import luigi
import numpy as np
import pytest

from synthesis_workflow.tasks import config
from synthesis_workflow.tools import get_layer_tags


DATA = Path(__file__).parent / "data"


def export_config(params, filepath):
    # Export params
    with open(filepath, "w") as configfile:
        params.write(configfile)


def reset_luigi_config(filepath):
    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read(filepath)

    config.reset_default_prefixes()

    yield luigi_config

    # Reset luigi config
    luigi_config.clear()


@pytest.fixture
def small_O1(tmp_path):
    """Dump a small O1 atlas in folder path"""
    atlas_dir = tmp_path / "small_O1"
    # fmt: off
    with open(os.devnull, "w") as f:
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
    br, _ = get_layer_tags(atlas_dir)
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
    """Change working directory before a test and change it back when the test is finished"""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


def get_config_parser(cfg_path):
    params = ConfigParser()
    params.read(cfg_path)
    return params


@pytest.fixture
def luigi_tools_params():
    params = ConfigParser()
    params.read_dict({"TaskA": {"a_cfg": "default_value_in_cfg"}})
    return params


@pytest.fixture
def small_O1_params():
    return get_config_parser(DATA / "in_small_O1" / "luigi.cfg")


@pytest.fixture
def vacuum_params():
    return get_config_parser(DATA / "in_vacuum" / "luigi.cfg")


def set_param_paths(params, tmp_working_dir, atlas_path=None):
    if atlas_path is not None:
        params["CircuitConfig"]["atlas_path"] = atlas_path.as_posix()
    params["BuildMorphsDF"]["neurondb_path"] = (
        DATA / "input_cells" / "neuronDB.xml"
    ).as_posix()
    params["BuildMorphsDF"]["morphology_dirs"] = json.dumps(
        {
            "repaired_morphology_path": (DATA / "input_cells").as_posix(),
        }
    )
    params["PathConfig"]["result_path"] = (tmp_working_dir / "out").as_posix()
    params["PathConfig"]["local_synthesis_input_path"] = (
        tmp_working_dir / "synthesis_input"
    ).as_posix()


@pytest.fixture
def luigi_tools_working_directory(tmp_working_dir, luigi_tools_params):
    # Setup config
    params = luigi_tools_params

    # Export config
    export_config(params, "luigi.cfg")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    yield tmp_working_dir

    # Reset luigi config
    luigi_config.clear()


@pytest.fixture
def vacuum_working_directory(tmp_working_dir, vacuum_params):
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
    shutil.copytree(DATA / "synthesis_input", tmp_working_dir / "synthesis_input")
    shutil.copyfile(DATA / "logging.conf", tmp_working_dir / "logging.conf")

    # Setup config
    params = small_O1_params
    set_param_paths(params, tmp_working_dir, small_O1)
    params["BuildAxonMorphologies"]["axon_cells_path"] = (
        DATA / "input_cells"
    ).as_posix()

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
