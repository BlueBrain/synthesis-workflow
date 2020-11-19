"""Luigi tasks for validation of synthesis."""
import json
import logging
from pathlib import Path

import luigi
import pandas as pd
import pkg_resources
import yaml
from atlas_analysis.planes.planes import load_planes_centerline
from neurom.view import view
from voxcell import VoxelData

from morphval import validation_main as morphval_validation
from synthesis_workflow.tasks.circuit import CreateAtlasLayerAnnotations
from synthesis_workflow.tasks.circuit import CreateAtlasPlanes
from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import MorphsDfLocalTarget
from synthesis_workflow.tasks.config import PathConfig
from synthesis_workflow.tasks.config import RunnerConfig
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.config import ValidationConfig
from synthesis_workflow.tasks.config import ValidationLocalTarget
from synthesis_workflow.tasks.luigi_tools import BoolParameter
from synthesis_workflow.tasks.luigi_tools import copy_params
from synthesis_workflow.tasks.luigi_tools import OptionalNumericalParameter
from synthesis_workflow.tasks.luigi_tools import ParamLink
from synthesis_workflow.tasks.luigi_tools import WorkflowError
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.synthesis import AddScalingRulesToParameters
from synthesis_workflow.tasks.synthesis import ApplySubstitutionRules
from synthesis_workflow.tasks.synthesis import BuildMorphsDF
from synthesis_workflow.tasks.synthesis import BuildSynthesisDistributions
from synthesis_workflow.tasks.synthesis import Synthesize
from synthesis_workflow.tasks.vacuum_synthesis import VacuumSynthesize
from synthesis_workflow.tools import load_circuit
from synthesis_workflow.vacuum_synthesis import VACUUM_SYNTH_MORPHOLOGY_PATH
from synthesis_workflow.validation import convert_mvd3_to_morphs_df
from synthesis_workflow.validation import parse_log
from synthesis_workflow.validation import plot_collage
from synthesis_workflow.validation import plot_density_profiles
from synthesis_workflow.validation import plot_morphometrics
from synthesis_workflow.validation import plot_path_distance_fits
from synthesis_workflow.validation import plot_scale_statistics
from synthesis_workflow.validation import plot_score_matrix
from synthesis_workflow.validation import SYNTH_MORPHOLOGY_PATH
from synthesis_workflow.validation import VacuumCircuit


L = logging.getLogger(__name__)


@copy_params(
    ext=ParamLink(PathConfig),
)
class ConvertMvd3(WorkflowTask):
    """Convert synthesize mvd3 file to morphs_df.csv file.

    Attributes:
        ext (str): Extension for morphology files.
    """

    def requires(self):
        """"""
        return Synthesize()

    def run(self):
        """"""
        synth_morphs_df = convert_mvd3_to_morphs_df(
            self.input()["out_mvd3"].path,
            self.input()["out_morphologies"].path,
            self.ext,
        )

        synth_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return MorphsDfLocalTarget(PathConfig().synth_morphs_df_path)


class PlotMorphometrics(WorkflowTask):
    """Plot cell morphometrics for two groups of cells so they can be easily compared.

    The generated images are like the following:

    .. image:: morphometrics-1.png
    """

    in_atlas = BoolParameter(default=False)
    """bool: Set to True to consider cells in an atlas."""

    config_features = luigi.DictParameter(default=None)
    """dict: The feature from ``morph_validator.feature_configs`` to plot."""

    morphometrics_path = luigi.Parameter(default="morphometrics")
    """str: Path to output directory (relative from ``PathConfig.result_path``)."""

    base_key = luigi.Parameter(default="repaired_morphology_path")
    """str: Base key to use in the morphology DataFrame."""

    comp_key = luigi.Parameter(default=SYNTH_MORPHOLOGY_PATH)
    """str: Compared key to use in the morphology DataFrame."""

    base_label = luigi.Parameter(default="bio")
    """str: Label for base morphologies."""

    comp_label = luigi.Parameter(default="synth")
    """str: Label for compared morphologies."""

    normalize = BoolParameter()
    """bool: Normalize data if set to True."""

    def requires(self):
        """"""
        if self.in_atlas:
            return {"morphs": BuildMorphsDF(), "mvd3": ConvertMvd3()}
        else:
            return {"vacuum": VacuumSynthesize(), "morphs": ApplySubstitutionRules()}

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input()["morphs"].path)
        if self.in_atlas:
            synth_morphs_df = pd.read_csv(self.input()["mvd3"].path)
            comp_key = self.comp_key
        else:
            synth_morphs_df = pd.read_csv(self.input()["vacuum"]["out_morphs_df"].path)
            comp_key = self.requires()["vacuum"].vacuum_synth_morphology_path

        plot_morphometrics(
            morphs_df,
            synth_morphs_df,
            self.output().path,
            base_key=self.base_key,
            comp_key=comp_key,
            base_label=self.base_label,
            comp_label=self.comp_label,
            normalize=self.normalize,
            config_features=self.config_features,
        )

    def output(self):
        """"""
        return ValidationLocalTarget(self.morphometrics_path)


@copy_params(
    nb_jobs=ParamLink(RunnerConfig),
    sample=ParamLink(ValidationConfig),
)
class PlotDensityProfiles(WorkflowTask):
    """Plot density profiles of neurites in an atlas.

    Attributes:
        sample (float): Number of cells to use. if None, use all available cells.
        nb_jobs (int) : Number of joblib workers.
    """

    density_profiles_path = luigi.Parameter(default="density_profiles.pdf")
    """str: Path for pdf file."""

    sample_distance = luigi.FloatParameter(default=10)
    """float: Distance between sampled points along neurites."""

    in_atlas = BoolParameter(default=False)
    """bool: Trigger atlas case."""

    def requires(self):
        """"""
        if self.in_atlas:
            return Synthesize()
        else:
            return VacuumSynthesize()

    def run(self):
        """"""
        if self.in_atlas:
            circuit = load_circuit(
                path_to_mvd3=self.input()["out_mvd3"].path,
                path_to_morphologies=self.input()["out_morphologies"].path,
                path_to_atlas=CircuitConfig().atlas_path,
            )
        else:
            df = pd.read_csv(self.input()["out_morphs_df"].path)
            circuit = VacuumCircuit(
                morphs_df=df,
                cells=pd.DataFrame(df["mtype"].unique(), columns=["mtypes"]),
                morphology_path=PathConfig().morphology_path,
            )

        plot_density_profiles(
            circuit,
            self.sample,
            self.in_atlas,
            self.sample_distance,
            self.output().path,
            self.nb_jobs,
        )

    def output(self):
        """"""
        return ValidationLocalTarget(self.density_profiles_path)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
    sample=ParamLink(ValidationConfig, default=20),
)
class PlotCollage(WorkflowTask):
    """Plot collage for all given mtypes.

    Collage reports show the cells in the atlas.

    The generated images are like the following:

    .. image:: collages-1.png

    Attributes:
        mtypes (list(str)): Mtypes to plot.
        nb_jobs (int): Number of joblib workers.
        joblib_verbose (int): Verbosity level of joblib.
        sample (float): Number of cells to use, if None, all available.
    """

    collage_base_path = luigi.Parameter(default="collages")
    """str: Path to the output folder."""

    dpi = luigi.IntParameter(default=1000)
    """int: Dpi for pdf rendering (rasterized)."""

    realistic_diameters = BoolParameter(
        default=True,
        description="Set or unset realistic diameter when NeuroM plot neurons",
    )
    """bool: Set or unset realistic diameter when NeuroM plot neurons."""

    linewidth = luigi.NumericalParameter(
        default=0.1,
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description="Linewidth used by NeuroM to plot neurons",
    )
    """float: Linewidth used by NeuroM to plot neurons."""

    diameter_scale = OptionalNumericalParameter(
        default=view._DIAMETER_SCALE,  # pylint: disable=protected-access
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description="Diameter scale used by NeuroM to plot neurons",
    )
    """float: Diameter scale used by NeuroM to plot neurons."""

    def requires(self):
        """"""
        return ConvertMvd3()

    def run(self):
        """"""
        if self.mtypes is None:
            mtypes = sorted(pd.read_csv(self.input().path).mtype.unique())
        else:
            mtypes = self.mtypes

        for mtype in mtypes:
            yield PlotSingleCollage(
                collage_base_path=self.collage_base_path,
                mtype=mtype,
                sample=self.sample,
                nb_jobs=self.nb_jobs,
                joblib_verbose=self.joblib_verbose,
                dpi=self.dpi,
                realistic_diameters=self.realistic_diameters,
                linewidth=self.linewidth,
                diameter_scale=self.diameter_scale,
            )

    def output(self):
        """"""
        return ValidationLocalTarget(self.collage_base_path)


@copy_params(
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
    collage_base_path=ParamLink(PlotCollage),
    sample=ParamLink(ValidationConfig),
    dpi=ParamLink(PlotCollage),
    realistic_diameters=ParamLink(PlotCollage),
    linewidth=ParamLink(PlotCollage),
    diameter_scale=ParamLink(PlotCollage),
)
class PlotSingleCollage(WorkflowTask):
    """Plot collage for a single mtype.

    Attributes:
        nb_jobs (int): Number of joblib workers.
        joblib_verbose (int): Verbosity level of joblib.
        collage_base_path (str): Path to the output folder.
        sample (float): Number of cells to use, if None, all available.
        dpi (int): Dpi for pdf rendering (rasterized).
        realistic_diameters (bool): Set or unset realistic diameter when NeuroM plot neurons.
        linewidth (float): Linewidth used by NeuroM to plot neurons.
        diameter_scale (float): Diameter scale used by NeuroM to plot neurons.
    """

    mtype = luigi.Parameter()
    """str or list(str): The mtype(s) to plot."""

    def requires(self):
        """"""
        return {
            "synthesis": Synthesize(),
            "planes": CreateAtlasPlanes(),
            "layers": CreateAtlasLayerAnnotations(),
        }

    def run(self):
        """"""
        mvd3_path = self.input()["synthesis"]["out_mvd3"].path
        morphologies_path = self.input()["synthesis"]["out_morphologies"].path
        atlas_path = CircuitConfig().atlas_path
        L.debug("Load circuit mvd3 from %s", mvd3_path)
        L.debug("Load circuit morphologies from %s", morphologies_path)
        L.debug("Load circuit atlas from %s", atlas_path)
        circuit = load_circuit(
            path_to_mvd3=mvd3_path,
            path_to_morphologies=morphologies_path,
            path_to_atlas=atlas_path,
        )

        planes_path = self.input()["planes"].path
        L.debug("Load planes from %s", planes_path)
        planes = load_planes_centerline(planes_path)["planes"]

        layer_annotation_path = self.input()["layers"]["annotations"].path
        L.debug("Load layer annotations from %s", layer_annotation_path)
        layer_annotation = VoxelData.load_nrrd(layer_annotation_path)

        L.debug("Plot single collage")
        plot_collage(
            circuit,
            planes,
            layer_annotation,
            self.mtype,
            pdf_filename=self.output().path,
            sample=self.sample,
            nb_jobs=self.nb_jobs,
            joblib_verbose=self.joblib_verbose,
            dpi=self.dpi,
            plot_neuron_kwargs={
                "realistic_diameters": self.realistic_diameters,
                "linewidth": self.linewidth,
                "diameter_scale": self.diameter_scale,
            },
        )

    def output(self):
        """"""
        return ValidationLocalTarget(
            (Path(self.collage_base_path) / self.mtype).with_suffix(".pdf")
        )


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
)
class PlotScales(WorkflowTask):
    """Plot scales.

    Create images of scaling factors used when the cells are generated.

    The generated images are like the following:

    .. image:: scale_statistics-5.png

    Attributes:
        mtypes (list(str)): Mtypes to plot.
    """

    scales_base_path = luigi.Parameter(default="scales")
    """str: Path to the output folder."""

    log_files = luigi.OptionalParameter(default=None)
    """str: (optional) Directory containing log files to parse."""

    neuron_type_position_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Neurite type and position: (.*)"
    )
    """str: Regex used to find neuron type and position."""

    default_scale_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Default barcode scale: (.*)"
    )
    """str: Regex used to find default scales."""

    target_scale_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Target barcode scale: (.*)"
    )
    """str: Regex used to find target scales."""

    neurite_hard_limit_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Neurite hard limit rescaling: (.*)"
    )
    """str: Regex used to find neurite hard limits."""

    def requires(self):
        """"""
        return ConvertMvd3()

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input().path)
        if self.mtypes is None:
            mtypes = sorted(morphs_df.mtype.unique())
        else:
            mtypes = self.mtypes

        debug_scales = self.log_files
        if debug_scales is None:
            debug_scales = self.requires().input()["debug_scales"].path
            if debug_scales is None:
                raise WorkflowError(
                    "%s task: either a 'log_files' argument must be provided, either the "
                    "'Synthesize' task must be run with 'debug_region_grower_scales' set "
                    "to a valid directory path" % self.__class__.__name__
                )

        # Plot statistics
        scale_data = parse_log(
            debug_scales,
            self.neuron_type_position_regex,
            self.default_scale_regex,
            self.target_scale_regex,
            self.neurite_hard_limit_regex,
        )
        scale_data.sort_values("worker_task_id", inplace=True)
        scale_data.set_index("worker_task_id", inplace=True)

        morphs_df["x"] = morphs_df["x"].round(4)
        morphs_df["y"] = morphs_df["y"].round(4)
        morphs_df["z"] = morphs_df["z"].round(4)
        scale_data["x"] = scale_data["x"].round(4)
        scale_data["y"] = scale_data["y"].round(4)
        scale_data["z"] = scale_data["z"].round(4)
        failed = scale_data.loc[
            ~scale_data.x.isin(morphs_df.x)
            & ~scale_data.y.isin(morphs_df.y)
            & ~scale_data.z.isin(morphs_df.z)
        ]
        L.info("Failed cells are %s", failed)
        plot_scale_statistics(
            mtypes,
            scale_data,
            output_dir=self.output().path,
        )

    def output(self):
        """"""
        return ValidationLocalTarget(self.scales_base_path)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    morphology_path=ParamLink(PathConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class PlotPathDistanceFits(WorkflowTask):
    """Plot fits of path distances as functions of their projection.

    The generated images are like the following:

    .. image:: path_distance_fit-1.png

    Attributes:
        mtypes (list(str)): Mtypes to plot.
        morphology_path (str): Column name to use in the DF from ApplySubstitutionRules.
        nb_jobs (int): Number of jobs.
    """

    output_path = luigi.Parameter(default="path_distance_fit.pdf")
    """str: Path to the output file."""

    outlier_percentage = luigi.IntParameter(default=90)
    """int: Percentage from which the outliers are removed."""

    def requires(self):
        """"""
        return {
            "scaling_rules": AddScalingRulesToParameters(),
            "rescaled": ApplySubstitutionRules(),
            "distributions": BuildSynthesisDistributions(),
        }

    def run(self):
        """"""

        L.debug("output_path = %s", self.output().path)
        plot_path_distance_fits(
            self.input()["scaling_rules"].path,
            self.input()["distributions"].path,
            self.input()["rescaled"].path,
            self.morphology_path,
            self.output().path,
            self.mtypes,
            self.outlier_percentage,
            self.nb_jobs,
        )

    def output(self):
        """"""
        return ValidationLocalTarget(self.output_path)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    morphology_path=ParamLink(PathConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class MorphologyValidationReports(WorkflowTask):
    """Create morphology validation reports.

    Attributes:
        mtypes (list(str)): List of mtypes to plot.
        morphology_path (str): Column name to use in the DF from ApplySubstitutionRules.
        nb_jobs (int): Number of jobs.
    """

    config_path = luigi.OptionalParameter(default=None)
    """str: (optional) Path to the configuration file. Use default configuration if not provided."""

    output_path = luigi.Parameter(default="morphology_validation_reports")
    """str: Path to the output file."""

    cell_figure_count = luigi.IntParameter(
        default=10, description="Number of example cells to show"
    )
    """int: Number of example cells to show"""

    bio_compare = BoolParameter(
        default=False, description="Use the bio compare template"
    )
    """bool: Use the bio compare template"""

    def requires(self):
        """"""
        return {
            "ref": ApplySubstitutionRules(),
            "test": ConvertMvd3(),
        }

    def run(self):
        """"""
        L.debug("Morphology validation output path = %s", self.output().path)

        ref_morphs_df = pd.read_csv(self.input()["ref"].path)
        test_morphs_df = pd.read_csv(self.input()["test"].path)

        if self.mtypes is not None:
            ref_morphs_df = ref_morphs_df.loc[ref_morphs_df["mtype"].isin(self.mtypes)]
            test_morphs_df = test_morphs_df.loc[
                test_morphs_df["mtype"].isin(self.mtypes)
            ]

        if self.config_path is not None:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
        else:
            with pkg_resources.resource_stream(
                "synthesis_workflow", "defaults/morphval_default_config.yaml"
            ) as f:
                default_config = yaml.safe_load(f)
                config = {
                    mtype: default_config["config"]["mtype"]
                    for mtype in ref_morphs_df["mtype"].unique()
                }

        ref_morphs_df = ref_morphs_df[["name", "mtype", self.morphology_path]].rename(
            columns={self.morphology_path: "filepath"}
        )
        test_morphs_df = test_morphs_df[
            ["name", "mtype", SYNTH_MORPHOLOGY_PATH]
        ].rename(columns={SYNTH_MORPHOLOGY_PATH: "filepath"})

        validator = morphval_validation.Validation(
            config,
            test_morphs_df,
            ref_morphs_df,
            self.output().ppath,
            create_timestamp_dir=False,
            notebook=False,
        )
        validator.validate_features(
            cell_figure_count=self.cell_figure_count, nb_jobs=self.nb_jobs
        )
        validator.write_report(validation_report=(not self.bio_compare))

    def output(self):
        """"""
        return ValidationLocalTarget(self.output_path)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    morphology_path=ParamLink(PathConfig),
    nb_jobs=ParamLink(RunnerConfig),
)
class PlotScoreMatrix(WorkflowTask):
    """Create score matrix reports.

    The generated images are like the following:

    .. image:: score_matrix_reports-1.png

    Attributes:
        mtypes (list(str)): List of mtypes to plot.
        morphology_path (str): Column name to use in the DF from ApplySubstitutionRules.
        nb_jobs (int): Number of jobs.
    """

    config_path = luigi.OptionalParameter(default=None)
    """str: (optional) Path to the configuration file. Use default configuration if not provided."""

    output_path = luigi.Parameter(default="score_matrix_reports.pdf")
    """str: Path to the output file."""

    in_atlas = BoolParameter(default=False)
    """bool: Trigger atlas case."""

    def requires(self):
        """"""
        if self.in_atlas:
            test_task = ConvertMvd3()
        else:
            test_task = VacuumSynthesize()
        return {
            "ref": ApplySubstitutionRules(),
            "test": test_task,
        }

    def run(self):
        """"""
        L.debug("Score matrix output path = %s", self.output().path)

        ref_morphs_df = pd.read_csv(self.input()["ref"].path)
        if self.in_atlas:
            test_morphs_df = pd.read_csv(self.input()["test"].path)
            file_path_col_name = SYNTH_MORPHOLOGY_PATH
        else:
            test_morphs_df = pd.read_csv(self.input()["test"]["out_morphs_df"].path)
            file_path_col_name = VACUUM_SYNTH_MORPHOLOGY_PATH

        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
        else:
            with pkg_resources.resource_stream(
                "synthesis_workflow", "defaults/score_matrix_default_config.json"
            ) as f:
                config = json.load(f)

        ref_morphs_df = ref_morphs_df[["name", "mtype", self.morphology_path]].rename(
            columns={self.morphology_path: "filepath"}
        )
        test_morphs_df = test_morphs_df[["name", "mtype", file_path_col_name]].rename(
            columns={file_path_col_name: "filepath"}
        )

        plot_score_matrix(
            ref_morphs_df,
            test_morphs_df,
            self.output().ppath,
            config,
            mtypes=self.mtypes,
            nb_jobs=self.nb_jobs,
        )

    def output(self):
        """"""
        return ValidationLocalTarget(self.output_path)
