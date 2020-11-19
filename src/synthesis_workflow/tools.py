"""utils functions."""
import glob
import json
import logging
import sys
import traceback
import warnings
from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import cpu_count

from bluepy.v2 import Circuit
from placement_algorithm.exceptions import SkipSynthesisError
from morph_tool.utils import neurondb_dataframe, find_morph
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import LocalAtlas

L = logging.getLogger(__name__)


def add_mtype_taxonomy(morphs_df, mtype_taxonomy):
    """From a dict with mtype to morph_class, fill in the morphs_df dataframe.

    Args:
        morphs_df (pandas.DataFrame): the morphs_df DataFrame
        mtype_taxonomy (pandas.DataFrame): with columns mtype and mClass
    """
    morphs_df["morph_class"] = morphs_df["mtype"].map(
        lambda mtype: mtype_taxonomy.loc[
            mtype_taxonomy.mtype == mtype, "mClass"
        ].to_list()[0]
    )
    return morphs_df


def add_morphology_paths(morphs_df, morphology_dirs):
    """Same as the path loader of morph_tool.utils.neurondb_dataframe, but add multiple columns.

    Args:
        morphs_df (pandas.DataFrame): the morphs_df DataFrame
        morphology_dirs: (dict) If passed, a column with the path to each morphology file
            will be added for each entry of the dict, where the column name is the dict key
    """
    for col_name, morphology_dir in morphology_dirs.items():
        f = partial(find_morph, Path(morphology_dir))
        morphs_df[col_name] = morphs_df["name"].apply(f)
    return morphs_df


def add_apical_points(morphs_df, apical_points):
    """Adds apical points isec in morphs_df.

    Args:
        morphs_df (pandas.DataFrame): the morphs_df DataFrame
        apical_points (dict): name of cell as key and apical point isec as value
    """
    morphs_df["apical_point_isec"] = -1
    # morphs_df["apical_point_isec_test"] = morphs_df["name"].map(apical_points)
    for name, apical_point in apical_points.items():
        morphs_df.loc[morphs_df.name == name, "apical_point_isec"] = apical_point
    morphs_df["apical_point_isec"] = morphs_df["apical_point_isec"].astype(int)
    return morphs_df


def load_neurondb_to_dataframe(
    neurondb_path, morphology_dirs=None, pc_in_types_path=None, apical_points_path=None
):
    """Loads morphology release to a dataframe.

    Args:
        neurondb_path (str): path to a neurondb.xml file
        morphology_dirs (dict): If passed, a column with the path to each morphology file
            will be added for each entry of the dict, where the column name is the dict key
        pc_in_types_path (str): path to mtype_taxonomy.tsv file
        apical_points_path (str): path to JSON file containing apical points
    """
    morphs_df = neurondb_dataframe(Path(neurondb_path))

    if morphology_dirs is not None:
        morphs_df = add_morphology_paths(morphs_df, morphology_dirs)

    if apical_points_path is not None:
        with open(apical_points_path, "r") as f:
            apical_points = json.load(f)
        morphs_df = add_apical_points(morphs_df, apical_points)

    if pc_in_types_path is not None:
        mtype_taxonomy = pd.read_csv(pc_in_types_path, sep="\t")
        morphs_df = add_mtype_taxonomy(morphs_df, mtype_taxonomy)

    return morphs_df


L = logging.getLogger(__name__)


def load_circuit(
    path_to_mvd3=None,
    path_to_morphologies=None,
    path_to_atlas=None,
    circuit_config=None,
):
    """Loads a circuit with bluepy.v2."""
    if circuit_config:
        return Circuit(circuit_config)
    return Circuit(
        {
            "cells": path_to_mvd3,
            "morphologies": path_to_morphologies,
            "atlas": path_to_atlas,
        }
    )


def ensure_dir(file_path):
    """Creates directory to save file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def find_case_insensitive_file(path):
    """Helper function to find a file ignoring case."""

    def either(c):
        return "[%s%s]" % (c.lower(), c.upper()) if c.isalpha() else c

    # Return the exact path if it exists
    exact_path = Path(path)
    if exact_path.exists():
        return exact_path

    directory = exact_path.parent

    # Try to find the file ignoring the case
    pattern = "".join(map(either, exact_path.name))
    complete_path = (directory / pattern).as_posix()

    all_possible_files = glob.glob(complete_path)

    if len(all_possible_files) == 1:
        found_file = all_possible_files[0]
        L.warning(
            "The file '%s' was not found, using '%s' instead.", exact_path, found_file
        )
        return found_file
    else:
        if len(all_possible_files) == 0:
            error_msg = (
                f"The file '{exact_path.as_posix()}' could not be found even "
                "ignoring case."
            )
        else:
            error_msg = (
                f"The file '{exact_path}' could not be found but several files were found "
                f"ignoring case: {all_possible_files}"
            )
        raise ValueError(error_msg)


def update_morphs_df(morphs_df_path, new_morphs_df):
    """Updates a morphs_df with new entries to preserve duplicates."""
    return pd.read_csv(morphs_df_path).merge(new_morphs_df, how="left")


def get_layer_tags(atlas_dir):
    """Creates a VoxelData with layer tags."""
    atlas = LocalAtlas(atlas_dir)

    names, ids = atlas.get_layers()  # pylint: disable=no-member
    br = VoxelData.load_nrrd(Path(atlas_dir) / "brain_regions.nrrd")
    layers = np.zeros_like(br.raw, dtype="uint8")
    layer_mapping = {}
    for layer_id, (ids_set, layer) in enumerate(zip(ids, names)):
        layer_mapping[layer_id] = layer
        layers[np.isin(br.raw, list(ids_set))] = layer_id + 1
    br.raw = layers
    return br, layer_mapping


class IdProcessingFormatter(logging.Formatter):
    """Logging formatter class."""

    def __init__(self, fmt=None, datefmt=None, current_id=None):
        """Cretate a new IdProcessingFormatter object."""
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s -- %(message)s"
        if datefmt is None:
            datefmt = "%Y/%m/%d %H:%M:%S"

        super().__init__(fmt, datefmt)
        self.orig_datefmt = datefmt
        self.orig_fmt = fmt
        if current_id is not None:
            self.set_id(current_id)

    def set_id(self, new_id):
        """Updates current ID to insert in format."""
        if new_id is None:
            new_fmt = self.orig_fmt
        else:
            msg_marker = "%(message)s"
            parts = self.orig_fmt.split(msg_marker, maxsplit=1)
            new_fmt = parts[0] + f"[WORKER TASK ID={new_id}] " + msg_marker + parts[1]
        super().__init__(new_fmt, self.orig_datefmt)


class DebugingFileHandler(logging.FileHandler):
    """Logging class that can be retrieved."""


def _wrap_worker(_id, worker, logger_kwargs=None):
    """Wraps the worker job and catch exceptions that must be caught."""
    try:
        if logger_kwargs is not None:
            logger_name = logger_kwargs.get("name")
            logger = logging.getLogger(logger_name)

            # Save current logger state
            old_level = (
                logger.level
                if logger.level != logging.NOTSET
                else logging.getLogger().level
            )
            old_propagate = logger.propagate

            # Set new logger state
            logger.setLevel(logger_kwargs.get("propagate", logging.DEBUG))
            logger.propagate = logger_kwargs.get("propagate", False)

            # Search old handlers
            handler_levels = {}
            file_handler = None
            for i in logger.handlers:
                if isinstance(i, DebugingFileHandler):
                    file_handler = i
                else:
                    handler_levels[i] = i.level
                    i.setLevel(old_level)

            # If no DebugingFileHandler was found
            if file_handler is None:
                # Setup file name
                log_file = logger_kwargs.get("log_file", worker.__class__.__name__)
                log_file = str(Path(log_file).with_suffix("") / f"scale-{_id}.log")
                ensure_dir(log_file)

                # Setup log formatter
                formatter = IdProcessingFormatter(
                    fmt=logger_kwargs.get("log_format"),
                    datefmt=logger_kwargs.get("date_format"),
                    current_id=_id,
                )

                # Setup handler
                file_handler = DebugingFileHandler(log_file)
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                logger.addHandler(file_handler)
            else:
                file_handler.formatter.set_id(_id)

        with warnings.catch_warnings():
            # Ignore all warnings in workers
            warnings.simplefilter("ignore")
            try:
                if logger_kwargs is not None:
                    logger.setLevel(logger_kwargs.get("log_level", logging.DEBUG))
                res = _id, worker(_id)
            finally:
                # Reset logger state
                logger.setLevel(old_level)
                logger.propagate = old_propagate

                # Reset old handler levels
                for i in logger.handlers:
                    if i in handler_levels:
                        i.setLevel(handler_levels[i])

        return res
    except SkipSynthesisError:
        return _id, None
    except Exception:  # pylint: disable=broad-except

        exception = "".join(traceback.format_exception(*sys.exc_info()))
        L.error("Task #%d failed with exception: %s", _id, exception)
        raise


def run_master(
    master_cls,
    kwargs,
    parser_args=None,
    defaults=None,
    nb_jobs=-1,
    verbose=10,
    logger_kwargs=None,
):
    """Runing the parallel computation, (adapted from placement_algorithm).

    Args:
        master_cls (class): The Master application
        kwargs (dict): A class with same attributes as CLI args
        parser_args (list): The arguments of the parser
        defaults (dict): The default values
        nb_jobs (int): Number of threads used
        verbose (int): verbosity level used by Joblib
        logger_kwargs (dict): Parameters given to logger in each processes
    """
    # Format keys to be compliant with argparse
    underscored = {k.replace("-", "_"): v for k, v in kwargs.items()}

    # Setup argument class
    if parser_args is None:
        parser_args = underscored.keys()
    SynthArgs = namedtuple("SynthArgs", parser_args)

    # Set default values
    if defaults is not None:
        for k, v in defaults.items():
            key = k.replace("-", "_")
            underscored[key] = underscored.get(key, v)

    # Build argument class instance
    args = SynthArgs(**underscored)
    L.info("Run %s with the following arguments: %s", master_cls.__name__, args)

    # Setup the worker
    master = master_cls()
    worker = master.setup(args)
    worker.setup(args)

    L.info("Running %d iterations.", len(master.task_ids))

    try:
        # Keep current log state to reset afterwards
        if logger_kwargs is not None:
            logger_name = logger_kwargs.get("name")
            logger = logging.getLogger(logger_name)
            handlers = set()
            for i in logger.handlers:
                handlers.add(i)

        # shuffle ids to speed up computation with uneven cell complexities
        task_ids = np.random.permutation(master.task_ids)

        # Run the worker
        L.info("Using batch size of %d tasks", int(len(task_ids) / cpu_count()))
        results = Parallel(
            n_jobs=nb_jobs,
            verbose=verbose,
            backend="multiprocessing",
            batch_size=1
            + int(len(task_ids) / (cpu_count() if nb_jobs == -1 else nb_jobs)),
        )(delayed(_wrap_worker)(i, worker, logger_kwargs) for i in task_ids)

        # Gather the results
        master.finalize(dict(results))

    finally:
        # This is usefull only when using joblib with 1 process
        if logger_kwargs is not None:
            for i in logger.handlers:
                if i not in handlers:
                    logger.removeHandler(i)
