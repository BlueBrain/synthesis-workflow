"""utils functions for luigi tasks."""
from pathlib import Path

import luigi
import pandas as pd


def load_circuit(
    path_to_mvd3=None,
    path_to_morphologies=None,
    path_to_atlas=None,
    circuit_config=None,
):
    """Load a circuit with bluepy.v2."""
    from bluepy.v2 import Circuit

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
    morphs_df_path, to_use_flag="all", morphology_path="morphology_path", h5_path=None,
):
    """Get valid morphs_df for diametrizer (select using flag + remove duplicates)."""
    morphs_df = pd.read_csv(morphs_df_path)
    if to_use_flag != "all":
        morphs_df = morphs_df[morphs_df[to_use_flag]]
    if h5_path is not None:
        morphs_df[morphology_path] = morphs_df[
            "_".join(morphology_path.split("_")[:-1])
        ].apply(lambda path: str((Path(h5_path) / Path(path).stem).with_suffix(".h5")))
    return morphs_df[["name", "mtype", morphology_path]].drop_duplicates()


def update_morphs_df(morphs_df_path, new_morphs_df):
    """Update a morphs_df with new entries to preserve duplicates."""
    return pd.read_csv(morphs_df_path).merge(new_morphs_df, how="left")


class diametrizerconfigs(luigi.Config):
    """Diametrizer configuration."""

    model = luigi.Parameter(
        config_path={"section": "DIAMETRIZER", "name": "model"}, default="generic",
    )
    terminal_threshold = luigi.FloatParameter(
        config_path={"section": "DIAMETRIZER", "name": "terminal_threshold"},
        default=2.0,
    )
    taper_min = luigi.FloatParameter(
        config_path={"section": "DIAMETRIZER", "name": "taper_min"}, default=-0.01,
    )
    taper_max = luigi.FloatParameter(
        config_path={"section": "DIAMETRIZER", "name": "taper_max"}, default=1e-6,
    )
    asymmetry_threshold_basal = luigi.FloatParameter(
        config_path={"section": "DIAMETRIZER", "name": "asymmetry_threshold_basal"},
        default=1.0,
    )
    asymmetry_threshold_apical = luigi.FloatParameter(
        config_path={"section": "DIAMETRIZER", "name": "asymmetry_threshold_apical"},
        default=0.2,
    )
    neurite_types = luigi.ListParameter(
        config_path={"section": "DIAMETRIZER", "name": "neurite_types"},
        default=["basal", "apical"],
    )

    trunk_max_tries = luigi.IntParameter(
        config_path={"section": "DIAMETRIZER", "name": "trunk_max_tries"}, default=100,
    )
    n_samples = luigi.IntParameter(
        config_path={"section": "DIAMETRIZER", "name": "n_samples"}, default=2,
    )

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.config_model = {
            "models": [self.model],
            "neurite_types": self.neurite_types,
            "terminal_threshold": self.terminal_threshold,
            "taper": {"max": self.taper_max, "min": self.taper_min},
            "asymmetry_threshold": {
                "apical": self.asymmetry_threshold_apical,
                "basal": self.asymmetry_threshold_basal,
            },
        }

        self.config_diametrizer = {
            "models": [self.model],
            "neurite_types": self.neurite_types,
            "asymmetry_threshold": {
                "apical": self.asymmetry_threshold_apical,
                "basal": self.asymmetry_threshold_basal,
            },
            "trunk_max_tries": self.trunk_max_tries,
            "n_samples": self.n_samples,
        }


class synthesisconfigs(luigi.Config):
    """Circuit configuration."""

    tmd_parameters_path = luigi.Parameter(
        config_path={"section": "SYNTHESIS", "name": "tmd_parameters_path"},
        default="tmd_parameters.json",
    )

    custom_tmd_parameters_path = luigi.Parameter(
        config_path={"section": "SYNTHESIS", "name": "custom_tmd_parameters_path"},
        default=None,
    )

    tmd_distributions_path = luigi.Parameter(
        config_path={"section": "SYNTHESIS", "name": "tmd_distributions_path"},
        default="tmd_distributions.json",
    )

    pc_in_types_path = luigi.Parameter(
        config_path={"section": "SYNTHESIS", "name": "pc_in_types_path"},
        default="pc_in_types.yaml",
    )

    cortical_thickness = luigi.Parameter(
        config_path={"section": "SYNTHESIS", "name": "cortical_thickness"},
        default="[165, 149, 353, 190, 525, 700]",
    )


class circuitconfigs(luigi.Config):
    """Circuit configuration."""

    circuit_somata_path = luigi.Parameter(
        config_path={"section": "CIRCUIT", "name": "circuit_somata_path"},
        default="circuit_somata.mvd3",
    )

    atlas_path = luigi.Parameter(
        config_path={"section": "CIRCUIT", "name": "atlas_path"}, default=None
    )


class pathconfigs(luigi.Config):
    """Morphology path configuration."""

    morphs_df_path = luigi.Parameter(
        config_path={"section": "PATHS", "name": "morphs_df_path"},
        default="morphs_df.csv",
    )

    synth_morphs_df_path = luigi.Parameter(
        config_path={"section": "PATHS", "name": "synth_morphs_df_path"},
        default="synth_morphs_df.csv",
    )

    synth_output_path = luigi.Parameter(
        config_path={"section": "PATHS", "name": "synth_output_path"},
        default="synthesized_morphologies",
    )

    substituted_morphs_df_path = luigi.Parameter(
        config_path={"section": "PATHS", "name": "substituted_morphs_df_path"},
        default="substituted_morphs_df.csv",
    )
