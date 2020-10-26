"""utils functions."""
import json
import logging
import random
import sys
import traceback
import warnings
from collections import namedtuple
from functools import partial
from pathlib import Path

import pandas as pd
from joblib import delayed
from joblib import Parallel

from bluepy.v2 import Circuit
from placement_algorithm.exceptions import SkipSynthesisError
from morph_tool.utils import neurondb_dataframe, find_morph


def add_mtype_taxonomy(morphs_df, mtype_taxonomy):
    """From a dict with mtype to morph_class, fill in the morphs_df dataframe.

    Args:
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
        morphology_dirs: (dict) If passed, a column with the path to each morphology file
            will be added for each entry of the dict, where the column name is the dict key
    """
    for col_name, morphology_dir in morphology_dirs.items():
        f = partial(find_morph, Path(morphology_dir))
        morphs_df[col_name] = morphs_df["name"].apply(f)
    return morphs_df


def add_apical_points(morphs_df, apical_points):
    """Add apical points isec in morphs_df.
    Args:
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
        morphology_dirs: (dict) If passed, a column with the path to each morphology file
            will be added for each entry of the dict, where the column name is the dict key
        mtype_taxonomy_path (str): path to mtype_taxonomy.tsv file
        apical_points (dict): name of cell as key and apical point isec as value
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
    """Load a circuit with bluepy.v2."""
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
    """Create directory to save file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def update_morphs_df(morphs_df_path, new_morphs_df):
    """Update a morphs_df with new entries to preserve duplicates."""
    return pd.read_csv(morphs_df_path).merge(new_morphs_df, how="left")


class IdProcessingFormatter(logging.Formatter):
    """Logging formatter class"""

    def __init__(self, fmt=None, datefmt=None, current_id=None):
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
        """Update current ID to insert in format"""
        if new_id is None:
            new_fmt = self.orig_fmt
        else:
            msg_marker = "%(message)s"
            parts = self.orig_fmt.split(msg_marker, maxsplit=1)
            new_fmt = parts[0] + f"[WORKER TASK ID={new_id}] " + msg_marker + parts[1]
        super().__init__(new_fmt, self.orig_datefmt)


class DebugingFileHandler(logging.FileHandler):
    """Logging class that can be retrieved"""


def _wrap_worker(_id, worker, logger_kwargs=None):
    """Wrap the worker job and catch exceptions that must be caught"""
    try:
        file_handler = None
        if logger_kwargs is not None:

            # Search old handlers
            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            for i in root.handlers:
                if isinstance(i, DebugingFileHandler):
                    file_handler = i
                    break

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
                root.addHandler(file_handler)
            else:
                file_handler.formatter.set_id(_id)

        with warnings.catch_warnings():
            # Ignore all warnings in workers
            warnings.simplefilter("ignore")
            try:
                old_level = root.level
                root.setLevel(logging.DEBUG)
                res = _id, worker(_id)
            finally:
                root.setLevel(old_level)

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
    """Runing the parrallel computation, (adapted from placement_algorithm).

    Args:
        master_cls: The Master application
        kwargs: A class with same attributes as CLI args
        parser_args: The arguments of the parser
        defaults: The default values
        nb_jobs: Number of threads used
        logger_kwargs: Parameters given to logger in each processes
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
        # Keep current log level to reset afterwards
        root = logging.getLogger()
        old_level = root.level
        if logger_kwargs is not None:
            root.setLevel(logging.DEBUG)

        # Run the worker
        random.shuffle(master.task_ids)
        results = Parallel(
            n_jobs=nb_jobs,
            verbose=verbose,
        )(delayed(_wrap_worker)(i, worker, logger_kwargs) for i in master.task_ids)

        # Gather the results
        master.finalize(dict(results))

    finally:
        # This is usefull only when using joblib with 1 process
        root.setLevel(old_level)
        for i in root.handlers:
            if isinstance(i, DebugingFileHandler):
                root.removeHandler(i)
