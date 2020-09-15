"""utils functions for luigi tasks."""
import logging
from pathlib import Path

import luigi
import pandas as pd


logger = logging.getLogger("luigi-interface")


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
        morphs_df = morphs_df.loc[morphs_df[to_use_flag]]
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

    model = luigi.Parameter(default="generic")
    terminal_threshold = luigi.FloatParameter(default=2.0)
    taper_min = luigi.FloatParameter(default=-0.01)
    taper_max = luigi.FloatParameter(default=1e-6)
    asymmetry_threshold_basal = luigi.FloatParameter(default=1.0)
    asymmetry_threshold_apical = luigi.FloatParameter(default=0.2)
    neurite_types = luigi.ListParameter(default=["basal", "apical"])

    trunk_max_tries = luigi.IntParameter(default=100)
    n_samples = luigi.IntParameter(default=2)

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

    tmd_parameters_path = luigi.Parameter(default="tmd_parameters.json")
    tmd_distributions_path = luigi.Parameter(default="tmd_distributions.json")
    pc_in_types_path = luigi.Parameter(default="pc_in_types.yaml")
    cortical_thickness = luigi.Parameter(default="[165, 149, 353, 190, 525, 700]")
    to_use_flag = luigi.Parameter(default="all")
    mtypes = luigi.ListParameter(default=["all"])


class circuitconfigs(luigi.Config):
    """Circuit configuration."""

    circuit_somata_path = luigi.Parameter(default="circuit_somata.mvd3")
    atlas_path = luigi.Parameter(default=None)


class pathconfigs(luigi.Config):
    """Morphology path configuration."""

    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    morphology_path = luigi.Parameter(default="morphology_path")
    synth_morphs_df_path = luigi.Parameter(default="synth_morphs_df.csv")
    synth_output_path = luigi.Parameter(default="synthesized_morphologies")
    substituted_morphs_df_path = luigi.Parameter(default="substituted_morphs_df.csv")


class BaseTask(luigi.Task):
    """Base class used to add customisable global parameters"""
    _global_configs = [diametrizerconfigs, synthesisconfigs, circuitconfigs, pathconfigs]

    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__(*args, **kwargs)

        # Replace run() method by a new one which adds logging before the actual run() method
        if not hasattr(self, "_actual_run"):
            # Rename actual run() method
            self._actual_run = super(BaseTask, self).__getattribute__("run")

            # Replace by _run_debug() method
            self.run = super(BaseTask, self).__getattribute__("_run_debug")

    def _run_debug(self):
        class_name = self.__class__.__name__
        logger.debug("Attributes of {} class after global processing:".format(class_name))
        for name, attr in self.get_params():
            try:
                logger.debug("Atribute: {} == {}".format(name, getattr(self, name)))
            except:
                logger.debug("Can't print '{}' attribute for unknown reason".format(name))

        logger.debug("Running {} task".format(class_name))
        gen = self._actual_run()
        logger.debug("{} task ended properly".format(class_name))
        return gen

    def __getattribute__(self, name):
        tmp = super(BaseTask, self).__getattribute__(name)
        if tmp is not None:
            return tmp
        for conf in self._global_configs:
            tmp_conf = conf()
            if hasattr(tmp_conf, name):
                return getattr(tmp_conf, name)
        return tmp


class BaseWrapperTask(BaseTask, luigi.WrapperTask):
    pass
