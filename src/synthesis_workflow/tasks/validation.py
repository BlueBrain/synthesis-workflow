"""Luigi tasks for validation of synthesis."""
import json
import logging
from pathlib import Path

import luigi
import pandas as pd
import pkg_resources
import yaml
from atlas_analysis.planes.planes import load_planes_centerline
from neurom.view import matplotlib_impl
from voxcell import CellCollection
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
from synthesis_workflow.tasks.luigi_tools import OptionalNumericalParameter
from synthesis_workflow.tasks.luigi_tools import ParamRef
from synthesis_workflow.tasks.luigi_tools import WorkflowError
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.luigi_tools import WorkflowWrapperTask
from synthesis_workflow.tasks.luigi_tools import copy_params
from synthesis_workflow.tasks.synthesis import AddScalingRulesToParameters
from synthesis_workflow.tasks.synthesis import ApplySubstitutionRules
from synthesis_workflow.tasks.synthesis import BuildMorphsDF
from synthesis_workflow.tasks.synthesis import BuildSynthesisDistributions
from synthesis_workflow.tasks.synthesis import Synthesize
from synthesis_workflow.tasks.vacuum_synthesis import VacuumSynthesize
from synthesis_workflow.tools import load_circuit
from synthesis_workflow.vacuum_synthesis import VACUUM_SYNTH_MORPHOLOGY_PATH
from synthesis_workflow.validation import SYNTH_MORPHOLOGY_PATH
from synthesis_workflow.validation import VacuumCircuit
from synthesis_workflow.validation import convert_mvd3_to_morphs_df
from synthesis_workflow.validation import get_debug_data
from synthesis_workflow.validation import plot_collage
from synthesis_workflow.validation import plot_density_profiles
from synthesis_workflow.validation import plot_morphometrics
from synthesis_workflow.validation import plot_path_distance_fits
from synthesis_workflow.validation import plot_scale_statistics
from synthesis_workflow.validation import plot_score_matrix

L = logging.getLogger(__name__)


@copy_params(
    ext=ParamRef(PathConfig),
)
class ConvertMvd3(WorkflowTask):
    """Convert synthesize mvd3 file to morphs_df.csv file.

    Attributes:
        ext (str): Extension for morphology files.
    """

    def requires(self):
        """ """
        return Synthesize()

    def run(self):
        """ """
        synth_morphs_df = convert_mvd3_to_morphs_df(
            self.input()["out_mvd3"].path,
            self.input()["out_morphologies"].path,
            self.ext,
        )

        synth_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return MorphsDfLocalTarget(PathConfig().synth_morphs_df_path)


class PlotMorphometrics(WorkflowTask):
    """Plot cell morphometrics for two groups of cells so they can be easily compared.

    The generated images are like the following:

    .. image:: morphometrics-1.png
    """

    in_atlas = BoolParameter(
        default=False, description=":bool: Set to True to consider cells in an atlas."
    )
    config_features = luigi.DictParameter(
        default=None,
        description=":dict: The features to plot.",
    )
    morphometrics_path = luigi.Parameter(
        default="morphometrics",
        description=":str: Path to output directory (relative from ``PathConfig.result_path``).",
    )
    base_key = luigi.Parameter(
        default="repaired_morphology_path",
        description=":str: Base key to use in the morphology DataFrame.",
    )
    comp_key = luigi.Parameter(
        default=SYNTH_MORPHOLOGY_PATH,
        description=":str: Compared key to use in the morphology DataFrame.",
    )
    base_label = luigi.Parameter(default="bio", description=":str: Label for base morphologies.")
    comp_label = luigi.Parameter(
        default="synth", description=":str: Label for compared morphologies."
    )
    normalize = BoolParameter(description=":bool: Normalize data if set to True.")

    def requires(self):
        """ """
        if self.in_atlas:
            return {"morphs": BuildMorphsDF(), "mvd3": ConvertMvd3()}
        else:
            return {"vacuum": VacuumSynthesize(), "morphs": ApplySubstitutionRules()}

    def run(self):
        """ """
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
        """ """
        return ValidationLocalTarget(self.morphometrics_path)


@copy_params(
    nb_jobs=ParamRef(RunnerConfig),
    sample=ParamRef(ValidationConfig),
)
class PlotDensityProfiles(WorkflowTask):
    """Plot density profiles of neurites in an atlas.

    Attributes:
        sample (float): Number of cells to use. if None, use all available cells.
        nb_jobs (int) : Number of joblib workers.
    """

    density_profiles_path = luigi.Parameter(
        default="density_profiles.pdf", description=":str: Path for pdf file."
    )
    sample_distance = luigi.FloatParameter(
        default=10,
        description=":float: Distance between sampled points along neurites.",
    )
    in_atlas = BoolParameter(default=False, description=":bool: Trigger atlas case.")

    def requires(self):
        """ """
        if self.in_atlas:
            return Synthesize()
        else:
            return VacuumSynthesize()

    def run(self):
        """ """
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
        """ """
        return ValidationLocalTarget(self.density_profiles_path)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
    nb_jobs=ParamRef(RunnerConfig),
    joblib_verbose=ParamRef(RunnerConfig),
    sample=ParamRef(ValidationConfig, default=20),
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

    collage_base_path = luigi.Parameter(
        default="collages", description=":str: Path to the output folder."
    )
    dpi = luigi.IntParameter(default=100, description=":int: Dpi for pdf rendering (rasterized).")
    realistic_diameters = BoolParameter(
        default=True,
        description=":bool: Set or unset realistic diameter when NeuroM plot neurons.",
    )
    linewidth = luigi.NumericalParameter(
        default=0.1,
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description=":float: Linewidth used by NeuroM to plot neurons.",
    )
    diameter_scale = OptionalNumericalParameter(
        default=matplotlib_impl._DIAMETER_SCALE,  # pylint: disable=protected-access
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description=":float: Diameter scale used by NeuroM to plot neurons.",
    )

    def requires(self):
        """ """
        return ConvertMvd3()

    def run(self):
        """ """
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
        """ """
        return ValidationLocalTarget(self.collage_base_path)


@copy_params(
    nb_jobs=ParamRef(RunnerConfig),
    joblib_verbose=ParamRef(RunnerConfig),
    collage_base_path=ParamRef(PlotCollage),
    sample=ParamRef(ValidationConfig),
    dpi=ParamRef(PlotCollage),
    realistic_diameters=ParamRef(PlotCollage),
    linewidth=ParamRef(PlotCollage),
    diameter_scale=ParamRef(PlotCollage),
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

    mtype = luigi.Parameter(description=":str: The mtype to plot.")

    def requires(self):
        """ """
        return {
            "synthesis": Synthesize(),
            "planes": CreateAtlasPlanes(),
            "layers": CreateAtlasLayerAnnotations(),
        }

    def run(self):
        """ """
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
        """ """
        return ValidationLocalTarget(
            (Path(self.collage_base_path) / self.mtype).with_suffix(".pdf")
        )


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
)
class PlotScales(WorkflowTask):
    """Plot scales.

    Create images of scaling factors used when the cells are generated.

    The generated images are like the following:

    .. image:: scale_statistics-5.png
      :width: 600

    Attributes:
        mtypes (list(str)): Mtypes to plot.
    """

    scales_base_path = luigi.Parameter(
        default="scales", description=":str: Path to the output folder."
    )
    debug_file = luigi.OptionalParameter(
        default=None,
        description=":str: File containing debug data.",
    )
    extra_stat_cols = luigi.ListParameter(
        default=tuple(),
        description=(
            ":str: The extra columns that should be ploted (these columns must exist in the debug "
            "data)."
        ),
    )

    def requires(self):
        """ """
        return ConvertMvd3()

    def run(self):
        """ """
        morphs_df = pd.read_csv(self.input().path)
        if self.mtypes is None:
            mtypes = sorted(morphs_df.mtype.unique())
        else:
            mtypes = self.mtypes

        debug_scales = self.debug_file
        if debug_scales is None:
            debug_scales = self.requires().input()["debug_scales"].path
            if debug_scales is None:
                raise WorkflowError(
                    "%s task: either a 'debug_file' argument must be provided, either the "
                    "'Synthesize' task must be run with 'debug_region_grower_scales' set "
                    "to a valid directory path" % self.__class__.__name__
                )

        # Plot statistics
        scale_data, stat_cols = get_debug_data(debug_scales)
        plotted_cols = list(self.extra_stat_cols) + stat_cols

        plot_scale_statistics(
            mtypes,
            scale_data,
            plotted_cols,
            output_dir=self.output().path,
        )

    def output(self):
        """ """
        return ValidationLocalTarget(self.scales_base_path)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
    morphology_path=ParamRef(PathConfig),
    nb_jobs=ParamRef(RunnerConfig),
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

    output_path = luigi.Parameter(
        default="path_distance_fit.pdf", description=":str: Path to the output file."
    )
    outlier_percentage = luigi.IntParameter(
        default=90, description=":int: Percentage from which the outliers are removed."
    )

    def requires(self):
        """ """
        return {
            "scaling_rules": AddScalingRulesToParameters(),
            "rescaled": ApplySubstitutionRules(),
            "distributions": BuildSynthesisDistributions(),
        }

    def run(self):
        """ """

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
        """ """
        return ValidationLocalTarget(self.output_path)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
    morphology_path=ParamRef(PathConfig),
    nb_jobs=ParamRef(RunnerConfig),
)
class MorphologyValidationReports(WorkflowTask):
    """Create morphology validation reports.

    Attributes:
        mtypes (list(str)): List of mtypes to plot.
        morphology_path (str): Column name to use in the DF from ApplySubstitutionRules.
        nb_jobs (int): Number of jobs.
    """

    config_path = luigi.OptionalParameter(
        default=None,
        description=(
            ":str: Path to the configuration file. Use default configuration if not provided."
        ),
    )
    output_path = luigi.Parameter(
        default="morphology_validation_reports",
        description=":str: Path to the output file.",
    )
    cell_figure_count = luigi.IntParameter(
        default=10, description=":int: Number of example cells to show."
    )
    bio_compare = BoolParameter(default=False, description=":bool: Use the bio compare template.")

    def requires(self):
        """ """
        return {
            "ref": ApplySubstitutionRules(),
            "test": ConvertMvd3(),
        }

    def run(self):
        """ """
        L.debug("Morphology validation output path = %s", self.output().path)

        ref_morphs_df = pd.read_csv(self.input()["ref"].path)
        test_morphs_df = pd.read_csv(self.input()["test"].path)

        if self.mtypes is not None:
            ref_morphs_df = ref_morphs_df.loc[ref_morphs_df["mtype"].isin(self.mtypes)]
            test_morphs_df = test_morphs_df.loc[test_morphs_df["mtype"].isin(self.mtypes)]

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
        test_morphs_df = test_morphs_df[["name", "mtype", SYNTH_MORPHOLOGY_PATH]].rename(
            columns={SYNTH_MORPHOLOGY_PATH: "filepath"}
        )

        validator = morphval_validation.Validation(
            config,
            test_morphs_df,
            ref_morphs_df,
            self.output().pathlib_path,
            create_timestamp_dir=False,
            notebook=False,
        )
        validator.validate_features(cell_figure_count=self.cell_figure_count, nb_jobs=self.nb_jobs)
        validator.write_report(validation_report=(not self.bio_compare))

    def output(self):
        """ """
        return ValidationLocalTarget(self.output_path)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
    morphology_path=ParamRef(PathConfig),
    nb_jobs=ParamRef(RunnerConfig),
)
class PlotScoreMatrix(WorkflowTask):
    """Create score matrix reports.

    The generated images are like the following:

    .. image:: score_matrix_reports-1.png
      :width: 400

    Attributes:
        mtypes (list(str)): List of mtypes to plot.
        morphology_path (str): Column name to use in the DF from ApplySubstitutionRules.
        nb_jobs (int): Number of jobs.
    """

    config_path = luigi.OptionalParameter(
        default=None,
        description=(
            ":str: Path to the configuration file. Use default configuration if not provided."
        ),
    )
    output_path = luigi.Parameter(
        default="score_matrix_reports.pdf", description=":str: Path to the output file."
    )
    in_atlas = BoolParameter(default=False, description=":bool: Trigger atlas case.")

    def requires(self):
        """ """
        if self.in_atlas:
            test_task = ConvertMvd3()
        else:
            test_task = VacuumSynthesize()
        return {
            "ref": ApplySubstitutionRules(),
            "test": test_task,
        }

    def run(self):
        """ """
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
            self.output().pathlib_path,
            config,
            mtypes=self.mtypes,
            nb_jobs=self.nb_jobs,
        )

    def output(self):
        """ """
        return ValidationLocalTarget(self.output_path)


class PlotRegionCollageFromCircuit(WorkflowWrapperTask):
    """Create a collage folder for all regions in regions list."""

    regions = luigi.ListParameter()

    def requires(self):
        """ """
        return [
            PlotCollageFromCircuit(region=region, collage_base_path=f"collages/{region}")
            for region in self.regions  # pylint: disable=not-an-iterable
        ]


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
    nb_jobs=ParamRef(RunnerConfig),
    sample=ParamRef(ValidationConfig),
    joblib_verbose=ParamRef(RunnerConfig),
)
class PlotCollageFromCircuit(WorkflowWrapperTask):
    """Plot collage for all given mtypes from a circuit (not used in this workflow).

    Attributes:
        mtypes (list(str)): Mtypes to plot.
        nb_jobs (int): Number of joblib workers.
        joblib_verbose (int): Verbosity level of joblib.
        sample (float): Number of cells to use, if None, all available.
    """

    collage_base_path = luigi.Parameter(
        default="collages", description=":str: Path to the output folder."
    )
    dpi = luigi.IntParameter(default=100, description=":int: Dpi for pdf rendering (rasterized).")
    realistic_diameters = BoolParameter(
        default=True,
        description=":bool: Set or unset realistic diameter when NeuroM plot neurons.",
    )
    linewidth = luigi.NumericalParameter(
        default=0.1,
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description=":float: Linewidth used by NeuroM to plot neurons.",
    )
    diameter_scale = OptionalNumericalParameter(
        default=matplotlib_impl._DIAMETER_SCALE,  # pylint: disable=protected-access
        var_type=float,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
        description=":float: Diameter scale used by NeuroM to plot neurons.",
    )
    circuit_config = luigi.Parameter()
    region = luigi.Parameter(default=None)

    def requires(self):
        """ """
        # pylint: disable=access-member-before-definition,attribute-defined-outside-init
        if self.mtypes is None:
            cells = CellCollection.load_sonata(self.circuit_config + "/circuit.morphologies.h5")
            self.mtypes = cells.as_dataframe().mtype.unique()
        tasks = []
        for mtype in self.mtypes:
            tasks.append(
                PlotSingleCollageFromCircuit(
                    collage_base_path=self.collage_base_path,
                    mtype=mtype,
                    sample=self.sample,
                    nb_jobs=self.nb_jobs,
                    joblib_verbose=self.joblib_verbose,
                    dpi=self.dpi,
                    realistic_diameters=self.realistic_diameters,
                    linewidth=self.linewidth,
                    diameter_scale=self.diameter_scale,
                    region=self.region,
                )
            )
        return tasks


@copy_params(
    nb_jobs=ParamRef(RunnerConfig),
    joblib_verbose=ParamRef(RunnerConfig),
    collage_base_path=ParamRef(PlotCollage),
    sample=ParamRef(ValidationConfig),
    dpi=ParamRef(PlotCollage),
    realistic_diameters=ParamRef(PlotCollage),
    linewidth=ParamRef(PlotCollage),
    diameter_scale=ParamRef(PlotCollage),
)
class PlotSingleCollageFromCircuit(WorkflowTask):
    """Plot collage for a single mtype to work with standalone PlotCollageFromCircuit.

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

    mtype = luigi.Parameter(description=":str: The mtype to plot.")
    with_y_field = luigi.BoolParameter(
        default=True, description=":str: Plot vector fields of orientations"
    )
    with_cells = luigi.BoolParameter(default=True, description=":str: Plot morphologies")
    hemisphere = luigi.Parameter(default="right")
    circuit_config = luigi.Parameter()
    region = luigi.Parameter(default=None)

    def requires(self):
        """ """
        return {
            "planes": CreateAtlasPlanes(region=self.region, hemisphere=self.hemisphere),
            "layers": CreateAtlasLayerAnnotations(),
        }

    def run(self):
        """ """
        circuit = load_circuit(
            path_to_mvd3=CircuitConfig().circuit_somata_path,
            path_to_morphologies=self.circuit_config + "morphologies",
            path_to_atlas=CircuitConfig().atlas_path,
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
            region=self.region,
            hemisphere=self.hemisphere,
            with_y_field=self.with_y_field,
            with_cells=self.with_cells,
        )

    def output(self):
        """ """
        return ValidationLocalTarget(
            (Path(self.collage_base_path) / self.mtype).with_suffix(".pdf")
        )
