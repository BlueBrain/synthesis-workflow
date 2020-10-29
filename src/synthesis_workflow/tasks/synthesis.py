"""Luigi tasks for morphology synthesis."""
import json
import logging
from functools import partial
from pathlib import Path

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
from ..synthesis import get_neurite_types
from ..synthesis import rescale_morphologies
from ..synthesis import run_synthesize_morphologies
from ..tools import ensure_dir
from ..tools import load_neurondb_to_dataframe
from .circuit import SliceCircuit
from .config import CircuitConfig
from .config import DiametrizerConfig
from .config import MorphsDfLocalTarget
from .config import PathConfig
from .config import RunnerConfig
from .config import SynthesisConfig
from .config import SynthesisLocalTarget
from .luigi_tools import BoolParameter
from .luigi_tools import copy_params
from .luigi_tools import ParamLink
from .luigi_tools import RatioParameter
from .luigi_tools import WorkflowTask
from .utils import GetSynthesisInputs


morphio.set_maximum_warnings(0)

L = logging.getLogger(__name__)


@copy_params(
    mtype_taxonomy_path=ParamLink(PathConfig),
)
class BuildMorphsDF(WorkflowTask):
    """Generate the list of morphologies with their mtypes and paths.

    Args:
        neurondb_path (str): path to the neuronDB file (XML)
        mtype_taxonomy_path (str): path to the mtype_taxonomy.tsv file
        morphology_dirs (str): dict (JSON format) in which keys are column names and values
            are the paths to each morphology file
        apical_points_path (str): path to the apical points file (JSON)
    """

    neurondb_path = luigi.Parameter(description="path to the neuronDB file (XML)")
    morphology_dirs = luigi.DictParameter(
        default=None,
        description=(
            "dict (JSON format) in which keys are column names and values are the paths to each "
            "morphology file"
        ),
    )
    apical_points_path = luigi.OptionalParameter(
        default=None, description="path to the apical points file (YAML)"
    )

    def requires(self):
        """"""
        return GetSynthesisInputs()

    def run(self):
        """"""

        L.debug("Build morphology dataframe from %s", self.neurondb_path)

        mtype_taxonomy_path = self.input().ppath / self.mtype_taxonomy_path
        morphs_df = load_neurondb_to_dataframe(
            self.neurondb_path,
            self.morphology_dirs,
            mtype_taxonomy_path,
            self.apical_points_path,
        )

        # Remove duplicated morphologies in L23
        morphs_df.drop_duplicates(subset=["name"], inplace=True)

        ensure_dir(self.output().path)
        morphs_df.to_csv(self.output().path)

    def output(self):
        """"""
        return MorphsDfLocalTarget(PathConfig().morphs_df_path)


class ApplySubstitutionRules(WorkflowTask):
    """Apply substitution rules to the morphology dataframe.

    Args:
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies
    """

    substitution_rules_path = luigi.Parameter(default="substitution_rules.yaml")

    def requires(self):
        """"""
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphs_df": BuildMorphsDF(),
        }

    def run(self):
        """"""
        substitution_rules_path = (
            self.input()["synthesis_input"].ppath / self.substitution_rules_path
        )
        with open(substitution_rules_path, "rb") as sub_file:
            substitution_rules = yaml.full_load(sub_file)

        substituted_morphs_df = apply_substitutions(
            pd.read_csv(self.input()["morphs_df"].path), substitution_rules
        )
        ensure_dir(self.output().path)
        substituted_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return MorphsDfLocalTarget(PathConfig().substituted_morphs_df_path)


@copy_params(
    tmd_parameters_path=ParamLink(SynthesisConfig),
)
class BuildSynthesisParameters(WorkflowTask):
    """Build the tmd_parameter.json for synthesis.

    Args:
        custom_tmd_parameters_path (str): custom path to input tmd_parameters. If not given,
            the one from
    """

    input_tmd_parameters_path = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphologies": ApplySubstitutionRules(),
        }

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input()["morphologies"].path)
        mtypes = morphs_df.mtype.unique()
        neurite_types = get_neurite_types(morphs_df, mtypes)

        if self.input_tmd_parameters_path is not None:
            L.info("Using custom tmd parameters")
            input_tmd_parameters_path = (
                self.input()["synthesis_input"].ppath / self.input_tmd_parameters_path
            )
            with open(input_tmd_parameters_path, "r") as f:
                custom_tmd_parameters = json.load(f)

        tmd_parameters = {}
        for mtype in mtypes:
            if self.input_tmd_parameters_path is None:
                tmd_parameters[mtype] = extract_input.parameters(
                    neurite_types=neurite_types[mtype],
                    diameter_parameters=DiametrizerConfig().config_diametrizer,
                )
            else:
                try:
                    tmd_parameters[mtype] = custom_tmd_parameters[mtype]
                except KeyError:
                    L.error("%s is not in the given tmd_parameter.json", mtype)
                    tmd_parameters[mtype] = {}
                tmd_parameters[mtype][
                    "diameter_params"
                ] = DiametrizerConfig().config_diametrizer
                tmd_parameters[mtype]["diameter_params"]["method"] = "external"

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4, sort_keys=True)

    def output(self):
        """"""
        return SynthesisLocalTarget(self.tmd_parameters_path)


@copy_params(
    morphology_path=ParamLink(PathConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class BuildSynthesisDistributions(WorkflowTask):
    """Build the tmd_distribution.json for synthesis.

    Args:
        morphology_path (str): column name in morphology dataframe to access morphology paths
    """

    def requires(self):
        """"""
        return ApplySubstitutionRules()

    def run(self):
        """"""
        L.debug("reading morphs df from: %s", self.input().path)
        morphs_df = pd.read_csv(self.input().path)

        mtypes = sorted(morphs_df.mtype.unique())
        L.debug("mtypes found: %s", mtypes)

        neurite_types = get_neurite_types(morphs_df, mtypes)
        L.debug("neurite_types found: %s", neurite_types)

        diameter_model_function = partial(
            build_diameter_models, config=DiametrizerConfig().config_model
        )

        tmd_distributions = build_distributions(
            mtypes,
            morphs_df,
            neurite_types,
            diameter_model_function,
            self.morphology_path,
            SynthesisConfig().cortical_thickness,
            nb_jobs=self.nb_jobs,
        )

        with self.output().open("w") as f:
            json.dump(tmd_distributions, f, cls=NumpyEncoder, indent=4, sort_keys=True)

    def output(self):
        """"""
        return SynthesisLocalTarget(SynthesisConfig().tmd_distributions_path)


class BuildAxonMorphsDF(BuildMorphsDF):
    """Generate the list of axon morphologies with their mtypes and paths."""

    axon_morphs_df_path = luigi.Parameter(default="axon_morphs_df.csv")

    def output(self):
        """"""
        return MorphsDfLocalTarget(self.axon_morphs_df_path)


class BuildAxonMorphologies(WorkflowTask):
    """Run choose-morphologies to synthesize axon morphologies.

    If no annotation file is given, axons will be randomly chosen from

    Args:
        axon_morphs_path (str): path to save .tsv file with list of morphologies for axon grafting
        annotations_path (str): path to annotations file used by
            ``placementAlgorithm.choose_morphologies``, if None, random axons will be choosen
        neurondb_basename (str): basename of the neurondb file for
            ``placementAlgorithm.choose_morphologies``
        axon_cells_path (str): path to the directory where cells with axons are located
        placement_rules_path (str): see ``placementAlgorithm.choose_morphologies``
        placement_alpha (float): see ``placementAlgorithm.choose_morphologies``
        placement_scales (list): see ``placementAlgorithm.choose_morphologies``
        placement_seed (int): see ``placementAlgorithm.choose_morphologies``
    """

    axon_morphs_path = luigi.Parameter(default="axon_morphs.tsv")
    annotations_path = luigi.Parameter(default=None)
    neurondb_basename = luigi.Parameter(
        default="neuronDB",
        description="base name of the neurondb file (without file extension)",
    )
    axon_cells_path = luigi.Parameter(
        description="path to the directory where cells with axons are located"
    )
    placement_rules_path = luigi.Parameter(default=None)
    placement_alpha = luigi.FloatParameter(default=1.0)
    placement_scales = luigi.ListParameter(default=None)
    placement_seed = luigi.IntParameter(default=0)
    nb_jobs = luigi.IntParameter(default=20)

    def requires(self):
        """"""
        tasks = {"circuit": SliceCircuit()}
        tasks["axon_cells"] = BuildAxonMorphsDF(
            neurondb_path=(
                Path(self.axon_cells_path) / self.neurondb_basename
            ).with_suffix(".xml"),
            morphology_dirs={"clone_path": self.axon_cells_path},
        )
        return tasks

    def run(self):
        """"""

        ensure_dir(self.output().path)
        if self.annotations_path is None:
            neurondb_path = None
            atlas_path = None
            axon_cells = self.input()["axon_cells"].path
        else:
            axon_cells = None
            neurondb_path = (
                Path(self.axon_cells_path) / self.neurondb_basename
            ).with_suffix(".dat")

        if any(
            [
                self.annotations_path is None,
                self.placement_rules_path is None,
                neurondb_path is None,
                axon_cells is not None,
            ]
        ):
            atlas_path = None
        else:
            atlas_path = CircuitConfig().atlas_path

        create_axon_morphologies_tsv(
            self.input()["circuit"].path,
            morphs_df_path=axon_cells,
            atlas_path=atlas_path,
            annotations_path=self.annotations_path,
            rules_path=self.placement_rules_path,
            morphdb_path=neurondb_path,
            alpha=self.placement_alpha,
            scales=self.placement_scales,
            seed=self.placement_seed,
            axon_morphs_path=self.output().path,
            nb_jobs=self.nb_jobs,
        )

    def output(self):
        """"""
        return MorphsDfLocalTarget(self.axon_morphs_path)


@copy_params(
    ext=ParamLink(PathConfig),
    morphology_path=ParamLink(PathConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class Synthesize(WorkflowTask):
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
    axon_morphs_base_dir = luigi.OptionalParameter(default=None)
    apical_points_path = luigi.Parameter(
        default="apical_points.yaml",
        description="path to the apical points file (YAML)",
    )
    debug_region_grower_scales = BoolParameter(default=False)
    max_drop_ratio = RatioParameter(
        default=0.1,
        description="The maximum drop ratio",
    )
    seed = luigi.IntParameter(default=0, description="pseudo-random generator seed")

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
        out_mvd3 = self.output()["out_mvd3"]
        out_morphologies = self.output()["out_morphologies"]
        out_apical_points = self.output()["apical_points"]
        debug_scales = self.output().get("debug_scales")

        ensure_dir(axon_morphs_path)
        ensure_dir(out_mvd3.path)
        ensure_dir(out_apical_points.path)
        ensure_dir(out_morphologies.path)

        # Get base-morph-dir argument value
        if self.axon_morphs_base_dir is None:
            axon_morphs_base_dir = get_axon_base_dir(
                pd.read_csv(self.requires()["axons"].input()["axon_cells"].path),
                "clone_path",
            )
        else:
            axon_morphs_base_dir = self.axon_morphs_base_dir

        L.debug("axon_morphs_base_dir = %s", axon_morphs_base_dir)

        # Build arguments for placement_algorithm.synthesize_morphologies.Master
        kwargs = {
            "cells_path": self.input()["circuit"].path,
            "tmd_parameters": self.input()["tmd_parameters"].path,
            "tmd_distributions": self.input()["tmd_distributions"].path,
            "atlas": CircuitConfig().atlas_path,
            "out_mvd3": out_mvd3.path,
            "out_apical": out_apical_points.path,
            "out_morph_ext": [str(self.ext)],
            "out_morph_dir": out_morphologies.path,
            "overwrite": True,
            "no_mpi": True,
            "morph-axon": axon_morphs_path,
            "base-morph-dir": axon_morphs_base_dir,
            "max_drop_ratio": self.max_drop_ratio,
            "seed": self.seed,
        }

        run_synthesize_morphologies(
            kwargs, nb_jobs=self.nb_jobs, debug_scales=debug_scales
        )

    def output(self):
        """"""
        outputs = {
            "out_mvd3": SynthesisLocalTarget(self.out_circuit_path),
            "out_morphologies": SynthesisLocalTarget(PathConfig().synth_output_path),
            "apical_points": SynthesisLocalTarget(self.apical_points_path),
        }
        if self.debug_region_grower_scales:
            outputs["debug_scales"] = SynthesisLocalTarget(
                PathConfig().debug_region_grower_scales_path
            )
        return outputs


@copy_params(
    morphology_path=ParamLink(PathConfig),
    tmd_parameters_path=ParamLink(SynthesisConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class AddScalingRulesToParameters(WorkflowTask):
    """Add scaling rules to tmd_parameter.json."""

    scaling_rules_path = luigi.Parameter(default="scaling_rules.yaml")

    def requires(self):
        """"""
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphologies": ApplySubstitutionRules(),
            "tmd_parameters": BuildSynthesisParameters(),
        }

    def run(self):
        """"""
        tmd_parameters = json.load(self.input()["tmd_parameters"].open("r"))

        if self.scaling_rules_path is not None:
            scaling_rules_path = (
                self.input()["synthesis_input"].ppath / self.scaling_rules_path
            )
            L.debug("Load scaling rules from %s", scaling_rules_path)
            scaling_rules = yaml.full_load(open(scaling_rules_path, "r"))
        else:
            scaling_rules = {}

        add_scaling_rules_to_parameters(
            tmd_parameters,
            self.input()["morphologies"].path,
            self.morphology_path,
            scaling_rules,
            self.nb_jobs,
        )

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4, sort_keys=True)

    def output(self):
        """"""
        return SynthesisLocalTarget(self.tmd_parameters_path)


@copy_params(
    morphology_path=ParamLink(PathConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class RescaleMorphologies(WorkflowTask):
    """Rescale morphologies for synthesis input."""

    rescaled_morphology_path = luigi.Parameter(default="rescaled_morphology_path")
    rescaled_morphology_base_path = luigi.Parameter(default="rescaled_morphologies")
    scaling_rules_path = luigi.Parameter(default="scaling_rules.yaml")
    rescaled_morphs_df_path = luigi.Parameter(default="rescaled_morphs_df.csv")
    scaling_mode = luigi.ChoiceParameter(default="y", choices=["y", "radial"])
    skip_rescale = BoolParameter(default=False)

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
            json.loads(SynthesisConfig().cortical_thickness),
            self.morphology_path,
            self.rescaled_morphology_base_path,
            self.rescaled_morphology_path,
            scaling_mode=self.scaling_mode,
            skip_rescale=self.skip_rescale,
        )

        rescaled_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return MorphsDfLocalTarget(self.rescaled_morphs_df_path)
