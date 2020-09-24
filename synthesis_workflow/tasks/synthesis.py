"""Luigi tasks for morphology synthesis."""
import json
import re
from functools import partial

import luigi
import pandas as pd
import yaml

import morphio
from diameter_synthesis.build_models import build as build_diameter_models
from region_grower.utils import NumpyEncoder
from tns import extract_input

from ..synthesis import add_scaling_rules_to_parameters
from ..synthesis import apply_substitutions
from ..synthesis import build_distributions
from ..synthesis import create_axon_morphologies_tsv
from ..synthesis import get_axon_base_dir
from ..synthesis import get_mean_neurite_lengths
from ..synthesis import get_neurite_types
from ..synthesis import rescale_morphologies
from ..synthesis import run_synthesize_morphologies
from ..tools import ensure_dir
from ..tools import get_morphs_df
from .circuit import SliceCircuit
from .config import circuitconfigs
from .config import diametrizerconfigs
from .config import logger as L
from .config import pathconfigs
from .config import synthesisconfigs
from .utils import BaseTask
from .utils import ExtParameter


morphio.set_maximum_warnings(0)


class ApplySubstitutionRules(luigi.Task):
    """Apply substitution rules to the morphology dataframe.

    Args:
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies
    """

    substitution_rules_path = luigi.Parameter(default="substitution_rules.yaml")

    def run(self):
        """"""
        with open(self.substitution_rules_path, "rb") as sub_file:
            substitution_rules = yaml.full_load(sub_file)

        substituted_morphs_df = apply_substitutions(
            pd.read_csv(pathconfigs().morphs_df_path), substitution_rules
        )
        ensure_dir(self.output().path)
        substituted_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(pathconfigs().substituted_morphs_df_path)


class BuildSynthesisParameters(BaseTask):
    """Build the tmd_parameter.json for synthesis.

    Args:
        to_use_flag (str): column name in morphology dataframe to select subset of cells
        custom_tmd_parameters_path (str): custom path to input tmd_parameters. If not given,
            the one from
    """

    to_use_flag = luigi.Parameter(default=None)
    input_tmd_parameters_path = luigi.Parameter(default=None)
    tmd_parameters_path = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""
        mtypes = list(
            set(get_morphs_df(self.input().path, to_use_flag=self.to_use_flag).mtype)
        )
        neurite_types = get_neurite_types(synthesisconfigs().pc_in_types_path, mtypes)

        if self.input_tmd_parameters_path is not None:
            L.info("Using custom tmd parameters")
            with open(self.input_tmd_parameters_path, "r") as f:
                custom_tmd_parameters = json.load(f)

        tmd_parameters = {}
        for mtype in mtypes:
            if self.input_tmd_parameters_path is None:
                tmd_parameters[mtype] = extract_input.parameters(
                    neurite_types=neurite_types[mtype],
                    diameter_parameters=diametrizerconfigs().config_diametrizer,
                )
            else:
                try:
                    tmd_parameters[mtype] = custom_tmd_parameters[mtype]
                except KeyError:
                    L.error("%s is not in the given tmd_parameter.json", mtype)
                    tmd_parameters[mtype] = {}
                tmd_parameters[mtype][
                    "diameter_params"
                ] = diametrizerconfigs().config_diametrizer
                tmd_parameters[mtype]["diameter_params"]["method"] = "external"

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4)

    def output(self):
        """"""
        return luigi.LocalTarget(self.tmd_parameters_path)


class BuildSynthesisDistributions(BaseTask):
    """Build the tmd_distribution.json for synthesis.

    Args:
        to_use_flag (str): column name in morphology dataframe to select subset of cells
        morphology_path (str): column name in morphology dataframe to access morphology paths
        h5_path (str): base path to h5 morphologies for TNS distribution extraction
    """

    to_use_flag = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default=None)
    h5_path = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""
        morphs_df = get_morphs_df(
            self.input().path,
            to_use_flag=self.to_use_flag,
            morphology_path=self.morphology_path,
            h5_path=self.h5_path,
        )
        mtypes = list(set(morphs_df.mtype))
        neurite_types = get_neurite_types(synthesisconfigs().pc_in_types_path, mtypes)

        diameter_model_function = partial(
            build_diameter_models, config=diametrizerconfigs().config_model
        )

        tmd_distributions = build_distributions(
            mtypes,
            morphs_df,
            neurite_types,
            diameter_model_function,
            self.morphology_path,
            synthesisconfigs().cortical_thickness,
        )

        with self.output().open("w") as f:
            json.dump(tmd_distributions, f, cls=NumpyEncoder, indent=4)

    def output(self):
        """"""
        return luigi.LocalTarget(synthesisconfigs().tmd_distributions_path)


class BuildSynthesisModels(luigi.WrapperTask):
    """Only build both json files for synthesis."""

    def requires(self):
        """"""
        return [BuildSynthesisParameters(), BuildSynthesisDistributions()]


class BuildAxonMorphologies(BaseTask):
    """Run choose-morphologies to synthesize axon morphologies.

    Args:
        out_circuit_path (str): path to circuit mvd3 with morphology data
        ext (str): extension for morphology files
        axon_morphs_path (str): path to .tsv file for axon grafting
        apical_points_path (str): path to .yaml file for recording apical points
    """

    axon_morphs_path = luigi.Parameter(default="axon_morphs.tsv")
    annotations_path = luigi.Parameter(default=None)
    rules_path = luigi.Parameter(default=None)
    morphdb_path = luigi.Parameter(default=None)
    placement_alpha = luigi.FloatParameter(default=1.0)
    placement_scales = luigi.ListParameter(default=None)
    placement_seed = luigi.IntParameter(default=0)
    nb_jobs = luigi.IntParameter(default=-1)

    def requires(self):
        """"""

        return {
            "substituted_cells": ApplySubstitutionRules(),
            "circuit": SliceCircuit(),
        }

    def run(self):
        """"""

        ensure_dir(self.output().path)

        atlas_path = circuitconfigs().atlas_path
        if any(
            [
                self.annotations_path is None,
                self.rules_path is None,
                self.morphdb_path is None,
            ]
        ):
            atlas_path = None

        create_axon_morphologies_tsv(
            self.input()["circuit"].path,
            self.input()["substituted_cells"].path,
            atlas_path=atlas_path,
            annotations_path=self.annotations_path,
            rules_path=self.rules_path,
            morphdb_path=self.morphdb_path,
            alpha=self.placement_alpha,
            scales=self.placement_scales,
            seed=self.placement_seed,
            axon_morphs_path=self.output().path,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.axon_morphs_path)


class Synthesize(BaseTask):
    """Run placement-algorithm to synthesize morphologies.

    Args:
        out_circuit_path (str): path to circuit mvd3 with morphology data
        ext (str): extension for morphology files
        axon_morphs_base_dir (str): base dir for morphology used for axon (.h5 files)
        apical_points_path (str): path to .yaml file for recording apical points
        morphology_path (str): column name to use in the DF to compute
            axon_morphs_base_dir if it is not provided
        nb_jobs (int): number of threads used for synthesis
    """

    out_circuit_path = luigi.Parameter(default="sliced_circuit_morphologies.mvd3")
    ext = ExtParameter(default="asc")
    axon_morphs_base_dir = luigi.OptionalParameter(default=None)
    apical_points_path = luigi.Parameter(default="apical_points.yaml")
    morphology_path = luigi.Parameter(default=None)
    nb_jobs = luigi.IntParameter(default=-1)

    def requires(self):
        """"""

        return {
            "substituted_cells": ApplySubstitutionRules(),
            "circuit": SliceCircuit(),
            "tmd_parameters": AddScalingRulesToParameters(),
            "tmd_distributions": BuildSynthesisDistributions(),
            "axons": BuildAxonMorphologies(),
        }

    def run(self):
        """"""

        axon_morphs_path = self.input()["axons"].path

        ensure_dir(axon_morphs_path)
        ensure_dir(self.output().path)
        ensure_dir(self.apical_points_path)
        ensure_dir(pathconfigs().synth_output_path)

        # Get base-morph-dir argument value
        if self.axon_morphs_base_dir is None:
            axon_morphs_base_dir = get_axon_base_dir(
                pd.read_csv(self.input()["substituted_cells"].path),
                self.morphology_path,
            )
        else:
            axon_morphs_base_dir = self.axon_morphs_base_dir

        L.debug("axon_morphs_base_dir = %s", axon_morphs_base_dir)

        # Build arguments for placement_algorithm.synthesize_morphologies.Master
        kwargs = {
            "cells_path": self.input()["circuit"].path,
            "tmd_parameters": self.input()["tmd_parameters"].path,
            "tmd_distributions": self.input()["tmd_distributions"].path,
            "atlas": circuitconfigs().atlas_path,
            "out_mvd3": self.output().path,
            "out_apical": self.apical_points_path,
            "out_morph_ext": [str(self.ext)],
            "out_morph_dir": pathconfigs().synth_output_path,
            "overwrite": True,
            "no_mpi": True,
            "morph-axon": axon_morphs_path,
            "base-morph-dir": axon_morphs_base_dir,
            "max_drop_ratio": str(0.1),
            "seed": str(0),
        }

        run_synthesize_morphologies(kwargs, nb_jobs=self.nb_jobs)

    def output(self):
        """"""
        return luigi.LocalTarget(self.out_circuit_path)


class GetMeanNeuriteLengths(BaseTask):
    """Get the mean neurite lenghts of a neuron population, per mtype and neurite type."""

    neurite_types = luigi.ListParameter(default=None)
    mtypes = luigi.ListParameter(default=None)
    morphology_path = luigi.Parameter(default=None)
    mean_lengths_path = luigi.Parameter(default="mean_neurite_lengths.yaml")
    percentile = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input().path)
        mean_lengths = {
            neurite_type: get_mean_neurite_lengths(
                morphs_df,
                neurite_type=neurite_type,
                mtypes=self.mtypes,
                morphology_path=self.morphology_path,
                percentile=self.percentile,
            )
            for neurite_type in self.neurite_types  # pylint: disable=not-an-iterable
        }

        L.info("Lengths for percentile=%s: %s", self.percentile, mean_lengths)

        with self.output().open("w") as f:
            yaml.dump(mean_lengths, f)

    def output(self):
        """"""
        return luigi.LocalTarget(self.mean_lengths_path)


class AddScalingRulesToParameters(BaseTask):
    """Add scaling rules to tmd_parameter.json."""

    scaling_rules_path = luigi.Parameter(default="scaling_rules.yaml")
    hard_limits_path = luigi.Parameter(default="hard_limits.yaml")
    tmd_parameters_path = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return {
            "tmd_parameters": BuildSynthesisParameters(),
            "mean_lengths": GetMeanNeuriteLengths(),
        }

    def run(self):
        """"""
        tmd_parameters = json.load(self.input()["tmd_parameters"].open("r"))
        mean_lengths = yaml.full_load(self.input()["mean_lengths"].open("r"))
        scaling_rules = yaml.full_load(open(self.scaling_rules_path, "r"))

        def _get_target_layer(target_layer_str):
            if re.match("^L", target_layer_str):
                position = 0.5
                if len(target_layer_str) == 3:
                    position = 1
                return int(target_layer_str[1]), position
            raise Exception("Scaling rule not understood: " + str(target_layer_str))

        for neurite_type in mean_lengths:
            for mtype, mean_length in mean_lengths[neurite_type].items():
                if (
                    mtype in scaling_rules
                    and scaling_rules[mtype] is not None
                    and neurite_type in scaling_rules[mtype]
                ):
                    tmd_parameters[mtype][neurite_type][
                        "expected_max_length"
                    ] = mean_length
                    layer, position = _get_target_layer(
                        scaling_rules[mtype][neurite_type]
                    )
                    tmd_parameters[mtype][neurite_type]["target_layer"] = layer
                    tmd_parameters[mtype][neurite_type][
                        "target_layer_position"
                    ] = position

        hard_limits = yaml.full_load(open(self.hard_limits_path, "r"))
        add_scaling_rules_to_parameters(
            tmd_parameters, mean_lengths, scaling_rules, hard_limits
        )

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4)

    def output(self):
        """"""
        return luigi.LocalTarget(self.tmd_parameters_path)


class RescaleMorphologies(BaseTask):
    """Rescale morphologies for synthesis input."""

    morphology_path = luigi.Parameter(default=None)
    rescaled_morphology_path = luigi.Parameter(default="rescaled_morphology_path")
    rescaled_morphology_base_path = luigi.Parameter(default="rescaled_morphologies")
    scaling_rules_path = luigi.Parameter(default="scaling_rules.yaml")
    rescaled_morphs_df_path = luigi.Parameter(default="rescaled_morphs_df.csv")
    scaling_mode = luigi.ChoiceParameter(default="y", choices=["y", "radial"])
    skip_rescale = luigi.BoolParameter(default=False)

    def requires(self):
        """"""
        return ApplySubstitutionRules()

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input().path)
        scaling_rules = yaml.full_load(open(self.scaling_rules_path, "r"))
        rescaled_morphs_df = rescale_morphologies(
            morphs_df,
            scaling_rules,
            json.loads(synthesisconfigs().cortical_thickness),
            self.morphology_path,
            self.rescaled_morphology_base_path,
            self.rescaled_morphology_path,
            scaling_mode=self.scaling_mode,
            skip_rescale=self.skip_rescale,
        )

        rescaled_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(self.rescaled_morphs_df_path)
