"""utils functions."""
import logging
import sys
import traceback
import warnings
from collections import namedtuple
from pathlib import Path

import pandas as pd
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from bluepy.v2 import Circuit
from placement_algorithm.exceptions import SkipSynthesisError


L = logging.getLogger("luigi-interface")


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


def get_morphs_df(
    morphs_df_path,
    to_use_flag="all",
    morphology_path="morphology_path",
    h5_path=None,
):
    """Get valid morphs_df for diametrizer (select using flag + remove duplicates)."""
    morphs_df = pd.read_csv(morphs_df_path)
    if to_use_flag != "all":
        morphs_df = morphs_df.loc[morphs_df[to_use_flag]]
    if h5_path is not None:
        morphs_df[morphology_path] = morphs_df[
            "_".join(morphology_path.split("_")[:-1])
        ].apply(lambda path: str((Path(h5_path) / Path(path).stem).with_suffix(".h5")))
    return morphs_df[["name", "mtype", morphology_path]].drop_duplicates()


def update_morphs_df(morphs_df_path, new_morphs_df):
    """Update a morphs_df with new entries to preserve duplicates."""
    return pd.read_csv(morphs_df_path).merge(new_morphs_df, how="left")


def _wrap_worker(_id, worker):
    """Wrap the worker job and catch exceptions that must be caught"""
    try:
        with warnings.catch_warnings():
            # Ignore all warnings in workers
            warnings.simplefilter("ignore")
            res = _id, worker(_id)
        return res
    except SkipSynthesisError:
        return _id, None
    except Exception:  # pylint: disable=broad-except

        exception = "".join(traceback.format_exception(*sys.exc_info()))
        L.error("Task #%d failed with exception: %s", _id, exception)
        raise


def run_master(master_cls, kwargs, parser_args=None, defaults=None, nb_jobs=-1):
    """To-be-executed on master node (MPI_RANK == 0).

    Args:
        master_cls: The Master application
        kwargs: A class with same attributes as CLI args
        parser_args: The arguments of the parser
        defaults: The default values
        nb_jobs: Number of threads used
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

    # Run the worker
    out = Parallel(
        n_jobs=nb_jobs,
        batch_size=20,
        verbose=20,
        max_nbytes=None,
    )(delayed(_wrap_worker)(i, worker) for i in tqdm(master.task_ids))

    # Gather the results
    result = dict(out)
    master.finalize(result)
