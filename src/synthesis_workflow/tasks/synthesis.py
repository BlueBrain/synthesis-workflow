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

from synthesis_workflow.synthesis import add_scaling_rules_to_parameters
from synthesis_workflow.synthesis import apply_substitutions
from synthesis_workflow.synthesis import build_distributions
from synthesis_workflow.synthesis import create_axon_morphologies_tsv
from synthesis_workflow.synthesis import get_axon_base_dir
from synthesis_workflow.synthesis import get_neurite_types
from synthesis_workflow.synthesis import rescale_morphologies
from synthesis_workflow.synthesis import run_synthesize_morphologies
from synthesis_workflow.tasks.circuit import SliceCircuit
from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import DiametrizerConfig
from synthesis_workflow.tasks.config import MorphsDfLocalTarget
from synthesis_workflow.tasks.config import PathConfig
from synthesis_workflow.tasks.config import RunnerConfig
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.config import SynthesisLocalTarget
from synthesis_workflow.tasks.luigi_tools import BoolParameter
from synthesis_workflow.tasks.luigi_tools import copy_params
from synthesis_workflow.tasks.luigi_tools import ParamLink
from synthesis_workflow.tasks.luigi_tools import RatioParameter
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.utils import GetSynthesisInputs
from synthesis_workflow.tools import ensure_dir
from synthesis_workflow.tools import find_case_insensitive_file
from synthesis_workflow.tools import load_neurondb_to_dataframe


morphio.set_maximum_warnings(0)

L = logging.getLogger(__name__)


@copy_params(
    mtype_taxonomy_path=ParamLink(PathConfig),
)
class BuildMorphsDF(WorkflowTask):
    """Generate the list of morphologies with their mtypes and paths.

    Attributes:
        mtype_taxonomy_path (str): Path to the mtype_taxonomy.tsv file.
    """

    neurondb_path = luigi.Parameter(description="Path to the neuronDB file (XML).")
    """str: Path to the neuronDB file (XML)."""

    morphology_dirs = luigi.DictParameter(
        default=None,
        description=(
            "Dict (JSON format) in which keys are column names and values are the paths to each "
            "morphology file."
        ),
    )
    """dict: Dict (JSON format) in which keys are column names and values are the paths to each
    morphology file."""

    apical_points_path = luigi.OptionalParameter(
        default=None, description="Path to the apical points file (JSON)."
    )
    """str: Path to the apical points file (JSON)."""

    def requires(self):
        """"""
        return GetSynthesisInputs()

    def run(self):
        """"""

        neurondb_path = find_case_insensitive_file(self.neurondb_path)

        L.debug("Build morphology dataframe from %s", neurondb_path)

        mtype_taxonomy_path = self.input().ppath / self.mtype_taxonomy_path
        morphs_df = load_neurondb_to_dataframe(
            neurondb_path,
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
    """Apply substitution rules to the morphology dataframe."""

    substitution_rules_path = luigi.Parameter(
        default="substitution_rules.yaml",
        description=(
            "Path to the file containing the rules to assign duplicated mtypes to morphologies."
        ),
    )
    """str: Path to the file containing the rules to assign duplicated mtypes to morphologies."""

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

    Attributes:
        tmd_parameters_path (str): The path to the TMD parameters.
    """

    input_tmd_parameters_path = luigi.Parameter(
        default=None,
        description=(
            "Custom path to input tmd_parameters. If not given, take the default"
            "parameters from TNS."
        ),
    )
    """str: Custom path to input tmd_parameters. If not given, take the default
    parameters from TNS."""

    def requires(self):
        """"""
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphologies": ApplySubstitutionRules(),
        }

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input()["morphologies"].path)
        mtypes = sorted(morphs_df.mtype.unique())
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

    Attributes:
        morphology_path (str): Column name in the morphology dataframe to access morphology paths.
        nb_jobs (int): Number of workers.
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

    axon_morphs_df_path = luigi.Parameter(
        default="axon_morphs_df.csv",
        description="Path to the CSV file containing axon morphologies.",
    )
    """str: Path to the CSV file containing axon morphologies."""

    def output(self):
        """"""
        return MorphsDfLocalTarget(self.axon_morphs_df_path)


class BuildAxonMorphologies(WorkflowTask):
    """Run choose-morphologies to synthesize axon morphologies.

    If no annotation file is given, axons will be randomly chosen from input cells.
    """

    axon_morphs_path = luigi.Parameter(
        default="axon_morphs.tsv",
        description="Path to save .tsv file with list of morphologies for axon grafting.",
    )
    """str: Path to save .tsv file with list of morphologies for axon grafting."""

    annotations_path = luigi.Parameter(
        default=None,
        description=(
            "Path to annotations file used by ``placementAlgorithm.choose_morphologies``. "
            "If None, random axons will be choosen."
        ),
    )
    """str: Path to annotations file used by ``placementAlgorithm.choose_morphologies``.
    If None, random axons will be choosen."""

    neurondb_basename = luigi.Parameter(
        default="neuronDB",
        description="Base name of the neurondb file (without file extension).",
    )
    """str: Basename of the neurondb file for ``placementAlgorithm.choose_morphologies``."""

    axon_cells_path = luigi.Parameter(
        description="Path to the directory where cells with axons are located."
    )
    """str: Path to the directory where cells with axons are located."""

    placement_rules_path = luigi.Parameter(
        default=None, description="See ``placementAlgorithm.choose_morphologies``."
    )
    """str: See ``placementAlgorithm.choose_morphologies``."""

    placement_alpha = luigi.FloatParameter(
        default=1.0, description="See ``placementAlgorithm.choose_morphologies``."
    )
    """float: See ``placementAlgorithm.choose_morphologies``."""

    placement_scales = luigi.ListParameter(
        default=None, description="See ``placementAlgorithm.choose_morphologies``."
    )
    """list: See ``placementAlgorithm.choose_morphologies``."""

    placement_seed = luigi.IntParameter(
        default=0, description="See ``placementAlgorithm.choose_morphologies``."
    )
    """int: See ``placementAlgorithm.choose_morphologies``."""

    nb_jobs = luigi.IntParameter(default=20, description="Number of workers.")
    """int: Number of workers."""

    def get_neuron_db_path(self, ext):
        """Helper function to fix neuronDB vs neurondb in file names."""
        return (Path(self.axon_cells_path) / self.neurondb_basename).with_suffix(
            "." + ext
        )

    def requires(self):
        """"""
        tasks = {"circuit": SliceCircuit()}

        neurondb_path = self.get_neuron_db_path("xml")

        tasks["axon_cells"] = BuildAxonMorphsDF(
            neurondb_path=neurondb_path,
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
            neurondb_path = find_case_insensitive_file(self.get_neuron_db_path("dat"))

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

    Attributes:
        ext (str): Extension for morphology files
        morphology_path (str): Column name to use in the DF to compute
            axon_morphs_base_dir if it is not provided
        nb_jobs (int): Number of threads used for synthesis
    """

    out_circuit_path = luigi.Parameter(
        default="sliced_circuit_morphologies.mvd3",
        description="Path to circuit mvd3 with morphology data.",
    )
    """str: Path to circuit mvd3 with morphology data."""

    axon_morphs_base_dir = luigi.OptionalParameter(
        default=None, description="Base dir for morphology used for axon (.h5 files)."
    )
    """str: Base dir for morphology used for axon (.h5 files)."""

    apical_points_path = luigi.Parameter(
        default="apical_points.yaml",
        description="Path to the apical points file (YAML).",
    )
    """str: Path to the apical points file (YAML)."""

    debug_region_grower_scales = BoolParameter(
        default=False,
        description="Trigger the recording of scaling factors computed by region-grower.",
    )
    """bool: Trigger the recording of scaling factors computed by region-grower."""

    max_drop_ratio = RatioParameter(
        default=0.1,
        description="The maximum drop ratio.",
    )
    """float: The maximum drop ratio."""

    seed = luigi.IntParameter(default=0, description="Pseudo-random generator seed.")
    """int: Pseudo-random generator seed."""

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
        if debug_scales is not None:
            debug_scales_path = debug_scales.path
        else:
            debug_scales_path = None

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
            kwargs, nb_jobs=self.nb_jobs, debug_scales_path=debug_scales_path
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
    """Add scaling rules to tmd_parameter.json.

    Attributes:
        morphology_path (str): Column name to use in the DF to compute
            axon_morphs_base_dir if it is not provided.
        tmd_parameters_path (str): The path to the TMD parameters.
        nb_jobs (int): Number of threads used for synthesis.
    """

    scaling_rules_path = luigi.Parameter(
        default="scaling_rules.yaml",
        description="Path to the file containing the scaling rules.",
    )
    """str: Path to the file containing the scaling rules."""

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
    """Rescale morphologies.

    Attributes:
        morphology_path (str): Column name to use in the DF to compute
            axon_morphs_base_dir if it is not provided.
        nb_jobs (int): Number of threads used for synthesis.
    """

    rescaled_morphology_path = luigi.Parameter(
        default="rescaled_morphology_path",
        description="Column name with rescaled morphology paths in the morphology DataFrame.",
    )
    """str: Column name with rescaled morphology paths in the morphology DataFrame."""

    rescaled_morphology_base_path = luigi.Parameter(
        default="rescaled_morphologies",
        description="Base path to rescaled morphologies.",
    )
    """str: Base path to rescaled morphologies."""

    scaling_rules_path = luigi.Parameter(
        default="scaling_rules.yaml",
        description="Path to the file containing the scaling rules.",
    )
    """str: Path to the file containing the scaling rules."""

    rescaled_morphs_df_path = luigi.Parameter(
        default="rescaled_morphs_df.csv", description="Path to the CSV morphology file."
    )
    """str: Path to the CSV morphology file."""

    scaling_mode = luigi.ChoiceParameter(
        default="y",
        choices=["y", "radial"],
        description=(
            "Scaling mode used: cells are either rescaled only according the Y axis or all axes."
        ),
    )
    """str: Scaling mode used: cells are either rescaled only according the Y axis or all axes."""

    skip_rescale = BoolParameter(
        default=False, description="Just copy input cells to the output directory."
    )
    """bool: Just copy input cells to the output directory."""

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
