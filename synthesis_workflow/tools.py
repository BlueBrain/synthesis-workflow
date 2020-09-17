"""utils functions."""
from pathlib import Path

import pandas as pd

from bluepy.v2 import Circuit


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
