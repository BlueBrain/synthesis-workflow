"""Luigi tasks for morphology synthesis."""
import json
import logging
from functools import partial
from pathlib import Path

import luigi
import morphio
import pandas as pd
import yaml
from diameter_synthesis.build_models import build as build_diameter_models
from region_grower.synthesize_morphologies import SynthesizeMorphologies
from region_grower.utils import NumpyEncoder
from tns import extract_input

from synthesis_workflow.synthesis import add_scaling_rules_to_parameters
from synthesis_workflow.synthesis import apply_substitutions
from synthesis_workflow.synthesis import build_distributions
from synthesis_workflow.synthesis import create_axon_morphologies_tsv
from synthesis_workflow.synthesis import get_axon_base_dir
from synthesis_workflow.synthesis import get_neurite_types
from synthesis_workflow.synthesis import rescale_morphologies
from synthesis_workflow.tasks.circuit import SliceCircuit
from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import DiametrizerConfig
from synthesis_workflow.tasks.config import MorphsDfLocalTarget
from synthesis_workflow.tasks.config import PathConfig
from synthesis_workflow.tasks.config import RunnerConfig
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.config import SynthesisLocalTarget
from synthesis_workflow.tasks.luigi_tools import BoolParameter
from synthesis_workflow.tasks.luigi_tools import ParamRef
from synthesis_workflow.tasks.luigi_tools import RatioParameter
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.luigi_tools import copy_params
from synthesis_workflow.tasks.utils import CreateAnnotationsFile
from synthesis_workflow.tasks.utils import GetSynthesisInputs
from synthesis_workflow.tools import find_case_insensitive_file
from synthesis_workflow.tools import load_neurondb_to_dataframe

morphio.set_maximum_warnings(0)

L = logging.getLogger(__name__)


@copy_params(
    mtype_taxonomy_path=ParamRef(PathConfig),
)
class BuildMorphsDF(WorkflowTask):
    """Generate the list of morphologies with their mtypes and paths.

    Attributes:
        mtype_taxonomy_path (str): Path to the mtype_taxonomy.tsv file.
    """

    neurondb_path = luigi.Parameter(description=":str: Path to the neuronDB file (XML).")
    morphology_dirs = luigi.DictParameter(
        default=None,
        description=(":dict: mapping between column names and paths to each morphology file."),
    )
    apical_points_path = luigi.OptionalParameter(
        default=None, description=":str: Path to the apical points file (JSON)."
    )

    def requires(self):
        """ """
        return GetSynthesisInputs()

    def run(self):
        """ """

        neurondb_path = find_case_insensitive_file(self.neurondb_path)

        L.debug("Build morphology dataframe from %s", neurondb_path)

        mtype_taxonomy_path = self.input().pathlib_path / self.mtype_taxonomy_path
        morphs_df = load_neurondb_to_dataframe(
            neurondb_path,
            self.morphology_dirs,
            mtype_taxonomy_path,
            self.apical_points_path,
        )

        # Remove duplicated morphologies in L23
        morphs_df.drop_duplicates(subset=["name"], inplace=True)

        morphs_df.to_csv(self.output().path)

    def output(self):
        """ """
        return MorphsDfLocalTarget(PathConfig().morphs_df_path)


class ApplySubstitutionRules(WorkflowTask):
    """Apply substitution rules to the morphology dataframe."""

    substitution_rules_path = luigi.Parameter(
        default="substitution_rules.yaml",
        description=(
            ":str: Path to the file containing the rules to assign duplicated mtypes to "
            "morphologies."
        ),
    )

    def requires(self):
        """ """
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphs_df": BuildMorphsDF(),
        }

    def run(self):
        """ """
        substitution_rules_path = (
            self.input()["synthesis_input"].pathlib_path / self.substitution_rules_path
        )
        with open(substitution_rules_path, "rb") as sub_file:
            substitution_rules = yaml.full_load(sub_file)

        substituted_morphs_df = apply_substitutions(
            pd.read_csv(self.input()["morphs_df"].path), substitution_rules
        )
        substituted_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return MorphsDfLocalTarget(PathConfig().substituted_morphs_df_path)


@copy_params(
    tmd_parameters_path=ParamRef(SynthesisConfig),
)
class BuildSynthesisParameters(WorkflowTask):
    """Build the tmd_parameter.json for synthesis.

    Attributes:
        tmd_parameters_path (str): The path to the TMD parameters.
    """

    input_tmd_parameters_path = luigi.Parameter(
        default=None,
        description=(
            ":str: Custom path to input tmd_parameters. If not given, take the default"
            "parameters from TNS."
        ),
    )

    def requires(self):
        """ """
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphologies": ApplySubstitutionRules(),
        }

    def run(self):
        """ """
        morphs_df = pd.read_csv(self.input()["morphologies"].path)
        mtypes = sorted(morphs_df.mtype.unique())
        neurite_types = get_neurite_types(morphs_df, mtypes)

        if self.input_tmd_parameters_path is not None:
            L.info("Using custom tmd parameters")
            input_tmd_parameters_path = (
                self.input()["synthesis_input"].pathlib_path / self.input_tmd_parameters_path
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
                tmd_parameters[mtype]["diameter_params"] = DiametrizerConfig().config_diametrizer
                tmd_parameters[mtype]["diameter_params"]["method"] = "external"

        with self.output().open("w") as f:
            json.dump(tmd_parameters, f, cls=NumpyEncoder, indent=4, sort_keys=True)

    def output(self):
        """ """
        return SynthesisLocalTarget(self.tmd_parameters_path)


@copy_params(
    morphology_path=ParamRef(PathConfig),
    nb_jobs=ParamRef(RunnerConfig),
)
class BuildSynthesisDistributions(WorkflowTask):
    """Build the tmd_distribution.json for synthesis.

    Attributes:
        morphology_path (str): Column name in the morphology dataframe to access morphology paths.
        nb_jobs (int): Number of workers.
    """

    def requires(self):
        """ """
        return ApplySubstitutionRules()

    def run(self):
        """ """
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
        """ """
        return SynthesisLocalTarget(SynthesisConfig().tmd_distributions_path)


class BuildAxonMorphsDF(BuildMorphsDF):
    """Generate the list of axon morphologies with their mtypes and paths."""

    axon_morphs_df_path = luigi.Parameter(
        default="axon_morphs_df.csv",
        description=":str: Path to the CSV file containing axon morphologies.",
    )

    def output(self):
        """ """
        return MorphsDfLocalTarget(self.axon_morphs_df_path)


class BuildAxonMorphologies(WorkflowTask):
    """Run choose-morphologies to synthesize axon morphologies.

    If no annotation file is given, axons will be randomly chosen from input cells.
    """

    axon_morphs_path = luigi.Parameter(
        default="axon_morphs.tsv",
        description=":str: Path to save .tsv file with list of morphologies for axon grafting.",
    )
    annotations_path = luigi.Parameter(
        default=None,
        description=(
            ":str: Path to annotations file used by "
            "``placementAlgorithm.app.choose_morphologies``. "
            "If None, random axons will be choosen."
        ),
    )
    neurondb_basename = luigi.Parameter(
        default="neuronDB",
        description=":str: Base name of the neurondb file (without file extension).",
    )
    axon_cells_path = luigi.Parameter(
        description=":str: Path to the directory where cells with axons are located."
    )
    placement_rules_path = luigi.Parameter(
        default=None,
        description=":str: See ``placementAlgorithm.app.choose_morphologies``.",
    )
    placement_alpha = luigi.FloatParameter(
        default=1.0,
        description=":float: See ``placementAlgorithm.app.choose_morphologies``.",
    )
    placement_scales = luigi.ListParameter(
        default=None,
        description=":list: See ``placementAlgorithm.app.choose_morphologies``.",
    )
    placement_seed = luigi.IntParameter(
        default=0,
        description=":int: See ``placementAlgorithm.app.choose_morphologies``.",
    )
    with_scores = BoolParameter(default=False, description=":bool: Export morphology scores.")
    filter_axons = BoolParameter(
        default=False,
        description=(
            ":bool: Read the neuronDB.xml file, filter cell with use_axon=True and generate a new "
            "neurondb.dat that is then read by ``placementAlgorithm.app.choose_morphologies``."
        ),
    )
    bias_kind = luigi.ChoiceParameter(
        choices=["uniform", "linear", "gaussian"],
        default="linear",
        description=":str: Kind of bias used to penalize scores of rescaled morphologies.",
    )
    with_optional_scores = BoolParameter(
        default=True,
        description=":bool: Use or ignore optional rules for morphology choice.",
    )
    nb_jobs = luigi.IntParameter(default=20, description=":int: Number of workers.")

    def get_neuron_db_path(self, ext):
        """Helper function to fix neuronDB vs neurondb in file names."""
        return (Path(self.axon_cells_path) / self.neurondb_basename).with_suffix("." + ext)

    def requires(self):
        """ """
        tasks = {"circuit": SliceCircuit()}

        neurondb_path = self.get_neuron_db_path("xml")

        tasks["axon_cells"] = BuildAxonMorphsDF(
            neurondb_path=neurondb_path,
            morphology_dirs={"clone_path": self.axon_cells_path},
        )
        return tasks

    def run(self):
        """ """
        if self.annotations_path is None:
            annotations_file = None
            neurondb_path = None
            atlas_path = None
            axon_cells = self.input()["axon_cells"].path
        else:
            if Path(self.annotations_path).is_dir():
                annotations_file = SynthesisLocalTarget("annotations.json").pathlib_path
                yield CreateAnnotationsFile(
                    annotation_dir=self.annotations_path,
                    destination=annotations_file,
                )
            else:
                input_task_target = yield GetSynthesisInputs()
                annotations_file = input_task_target.pathlib_path / self.annotations_path
            axon_cells = None
            if self.filter_axons:
                neurondb_path = self.output()["axons_neurondb"].path
                input_neurondb_path = find_case_insensitive_file(self.get_neuron_db_path("xml"))
                morphs_df = load_neurondb_to_dataframe(input_neurondb_path)
                morphs_df.drop(morphs_df.loc[~morphs_df["use_axon"]].index, inplace=True)
                morphs_df.drop(columns=["use_axon"], inplace=True)
                morphs_df.to_csv(neurondb_path, sep=" ", header=False, index=False)
            else:
                neurondb_path = find_case_insensitive_file(self.get_neuron_db_path("dat"))

        if any(
            [
                annotations_file is None,
                self.placement_rules_path is None,
                neurondb_path is None,
                axon_cells is not None,
            ]
        ):
            atlas_path = None
        else:
            atlas_path = CircuitConfig().atlas_path

        with_scores = self.with_scores
        if with_scores:
            scores_output_folder = self.output()["scores"].path
        else:
            scores_output_folder = None

        create_axon_morphologies_tsv(
            self.input()["circuit"].path,
            morphs_df_path=axon_cells,
            atlas_path=atlas_path,
            annotations_path=annotations_file,
            rules_path=self.placement_rules_path,
            morphdb_path=neurondb_path,
            alpha=self.placement_alpha,
            scales=self.placement_scales,
            seed=self.placement_seed,
            axon_morphs_path=self.output()["morphs"].path,
            scores_output_path=scores_output_folder,
            bias_kind=self.bias_kind,
            with_optional_scores=self.with_optional_scores,
            nb_jobs=self.nb_jobs,
        )

    def output(self):
        """ """
        targets = {
            "morphs": MorphsDfLocalTarget(self.axon_morphs_path),
        }
        if self.with_scores:
            scores_output_folder = Path(self.axon_morphs_path).with_suffix("")
            targets["scores"] = MorphsDfLocalTarget(scores_output_folder)
        if self.filter_axons:
            targets["axons_neurondb"] = MorphsDfLocalTarget("axons_neurondb.dat")
        return targets


@copy_params(
    ext=ParamRef(PathConfig),
    morphology_path=ParamRef(PathConfig),
    nb_jobs=ParamRef(RunnerConfig),
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
        description=":str: Path to circuit mvd3 with morphology data.",
    )
    axon_morphs_base_dir = luigi.OptionalParameter(
        default=None,
        description=":str: Base dir for morphology used for axon (.h5 files).",
    )
    apical_points_path = luigi.Parameter(
        default="apical_points.yaml",
        description=":str: Path to the apical points file (YAML).",
    )
    debug_region_grower_scales = BoolParameter(
        default=False,
        description=":bool: Trigger the recording of scaling factors computed by region-grower.",
    )
    max_drop_ratio = RatioParameter(
        default=0.1,
        description=":float: The maximum drop ratio.",
    )
    apply_jitter = BoolParameter(
        default=False, description=":bool: Apply jitter to all sections of axons."
    )
    scaling_jitter_std = luigi.NumericalParameter(
        default=0.2,
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description=":float: The std value of the scaling jitter to apply.",
    )
    rotational_jitter_std = luigi.NumericalParameter(
        default=10,
        var_type=float,
        min_value=0,
        max_value=180,
        left_op=luigi.parameter.operator.lt,
        right_op=luigi.parameter.operator.le,
        description=":float: The std value of the scaling jitter to apply (in degrees).",
    )
    seed = luigi.IntParameter(default=0, description=":int: Pseudo-random generator seed.")

    def requires(self):
        """ """

        return {
            "substituted_cells": ApplySubstitutionRules(),
            "circuit": SliceCircuit(),
            "tmd_parameters": AddScalingRulesToParameters(),
            "tmd_distributions": BuildSynthesisDistributions(),
            "axons": BuildAxonMorphologies(),
        }

    def run(self):
        """ """

        axon_morphs_path = self.input()["axons"]["morphs"].path
        out_mvd3 = self.output()["out_mvd3"]
        out_morphologies = self.output()["out_morphologies"]
        out_apical_points = self.output()["apical_points"]
        debug_scales = self.output().get("debug_scales")

        if debug_scales is not None:
            debug_scales_path = debug_scales.path
        else:
            debug_scales_path = None

        # Get base-morph-dir argument value
        if self.axon_morphs_base_dir is None:
            axon_morphs_base_dir = get_axon_base_dir(
                pd.read_csv(self.requires()["axons"].input()["axon_cells"].path),
                "clone_path",
            )
        else:
            axon_morphs_base_dir = self.axon_morphs_base_dir

        L.debug("axon_morphs_base_dir = %s", axon_morphs_base_dir)

        # Build arguments for region_grower.synthesize_morphologies.SynthesizeMorphologies
        kwargs = {
            "input_cells": self.input()["circuit"].path,
            "tmd_parameters": self.input()["tmd_parameters"].path,
            "tmd_distributions": self.input()["tmd_distributions"].path,
            "atlas": CircuitConfig().atlas_path,
            "out_cells": out_mvd3.path,
            "out_apical": out_apical_points.path,
            "out_morph_ext": [str(self.ext)],
            "out_morph_dir": out_morphologies.path,
            "overwrite": True,
            "with_mpi": False,
            "morph_axon": axon_morphs_path,
            "base_morph_dir": axon_morphs_base_dir,
            "max_drop_ratio": self.max_drop_ratio,
            "seed": self.seed,
            "nb_processes": self.nb_jobs,
        }

        if self.apply_jitter:
            kwargs["scaling_jitter_std"] = self.scaling_jitter_std
            kwargs["rotational_jitter_std"] = self.rotational_jitter_std

        if debug_scales_path is not None:
            kwargs["out_debug_data"] = debug_scales_path

        synthesizer = SynthesizeMorphologies(**kwargs)
        synthesizer.synthesize()

    def output(self):
        """ """
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
    morphology_path=ParamRef(PathConfig),
    tmd_parameters_path=ParamRef(SynthesisConfig),
    nb_jobs=ParamRef(RunnerConfig),
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
        description=":str: Path to the file containing the scaling rules.",
    )

    def requires(self):
        """ """
        return {
            "synthesis_input": GetSynthesisInputs(),
            "morphologies": ApplySubstitutionRules(),
            "tmd_parameters": BuildSynthesisParameters(),
        }

    def run(self):
        """ """
        tmd_parameters = json.load(self.input()["tmd_parameters"].open("r"))

        if self.scaling_rules_path is not None:
            scaling_rules_path = (
                self.input()["synthesis_input"].pathlib_path / self.scaling_rules_path
            )
            L.debug("Load scaling rules from %s", scaling_rules_path)
            with open(scaling_rules_path, "r") as f:
                scaling_rules = yaml.full_load(f)
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
        """ """
        return SynthesisLocalTarget(self.tmd_parameters_path)


@copy_params(
    morphology_path=ParamRef(PathConfig),
    nb_jobs=ParamRef(RunnerConfig),
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
        description=":str: Column name with rescaled morphology paths in the morphology DataFrame.",
    )
    rescaled_morphology_base_path = luigi.Parameter(
        default="rescaled_morphologies",
        description=":str: Base path to rescaled morphologies.",
    )
    scaling_rules_path = luigi.Parameter(
        default="scaling_rules.yaml",
        description=":str: Path to the file containing the scaling rules.",
    )
    rescaled_morphs_df_path = luigi.Parameter(
        default="rescaled_morphs_df.csv",
        description=":str: Path to the CSV morphology file.",
    )
    scaling_mode = luigi.ChoiceParameter(
        default="y",
        choices=["y", "radial"],
        description=(
            ":str: Scaling mode used: cells are either rescaled only according the Y axis or all "
            "axes."
        ),
    )
    skip_rescale = BoolParameter(
        default=False,
        description=":bool: Just copy input cells to the output directory.",
    )

    def requires(self):
        """ """
        return ApplySubstitutionRules()

    def run(self):
        """ """
        morphs_df = pd.read_csv(self.input().path)
        with open(self.scaling_rules_path, "r") as f:
            scaling_rules = yaml.full_load(f)
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
        """ """
        return MorphsDfLocalTarget(self.rescaled_morphs_df_path)
