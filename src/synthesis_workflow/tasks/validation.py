"""Luigi tasks for validation of synthesis."""
import logging
from pathlib import Path

import luigi
import pandas as pd
import pkg_resources
import yaml
from voxcell import VoxelData
from atlas_analysis.planes.planes import load_planes_centerline

from morphval import validation_main as morphval_validation
from synthesis_workflow.tools import load_circuit
from synthesis_workflow.validation import convert_mvd3_to_morphs_df
from synthesis_workflow.validation import parse_log
from synthesis_workflow.validation import plot_collage
from synthesis_workflow.validation import plot_density_profiles
from synthesis_workflow.validation import plot_morphometrics
from synthesis_workflow.validation import plot_path_distance_fits
from synthesis_workflow.validation import plot_scale_statistics
from synthesis_workflow.validation import SYNTH_MORPHOLOGY_PATH
from synthesis_workflow.validation import VacuumCircuit
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
from synthesis_workflow.tasks.luigi_tools import ParamLink
from synthesis_workflow.tasks.luigi_tools import WorkflowError
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.synthesis import AddScalingRulesToParameters
from synthesis_workflow.tasks.synthesis import ApplySubstitutionRules
from synthesis_workflow.tasks.synthesis import BuildMorphsDF
from synthesis_workflow.tasks.synthesis import BuildSynthesisDistributions
from synthesis_workflow.tasks.synthesis import Synthesize
from synthesis_workflow.tasks.vacuum_synthesis import VacuumSynthesize


L = logging.getLogger(__name__)


@copy_params(
    ext=ParamLink(PathConfig),
)
class ConvertMvd3(WorkflowTask):
    """Convert synthesize mvd3 file to morphs_df.csv file.

    Args:
        ext (str): extension for morphology files
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
    """Plot morphometric.

    Args:
        in_atlas (bool): set to True to consider cells in an atlas
        config_features (dict): the feature from ``morph_validator.feature_configs`` to plot
        morphometrics_path (str): path to output directory (relative from PathConfig.result_path)
        base_key (str): base key to use in the morphology DataFrame
        comp_key (str): compared key to use in the morphology DataFrame
        base_label (str): label for base morphologies
        comp_label (str): label for compared morphologies
        normalize (str): normalize data if set to True
    """

    in_atlas = BoolParameter(default=False)
    config_features = luigi.DictParameter(default=None)
    morphometrics_path = luigi.Parameter(default="morphometrics")
    base_key = luigi.Parameter(default="repaired_morphology_path")
    comp_key = luigi.Parameter(default=SYNTH_MORPHOLOGY_PATH)
    base_label = luigi.Parameter(default="bio")
    comp_label = luigi.Parameter(default="synth")
    normalize = BoolParameter()

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

    Args:
        density_profiles_path (str): path for pdf file
        sample_distance (float): distance between sampled points along neurites
        sample (float): number of cells to use, if None, all available
        circuit_type (str): type of the circuit (in_vacuum, O1, etc...)
        nb_jobs (int) : number of joblib workers
    """

    density_profiles_path = luigi.Parameter(default="density_profiles.pdf")
    sample_distance = luigi.FloatParameter(default=10)
    in_atlas = BoolParameter(default=False)

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
    """Plot collage.

    Args:
        collage_base_path (str): path to the output folder
        sample (float): number of cells to use, if None, all available
        mtypes (list(str)): mtypes to plot
        nb_jobs (int) : number of joblib workers
        joblib_verbose (int) verbose level of joblib
        dpi (int): dpi for pdf rendering (rasterized)

    """

    collage_base_path = luigi.Parameter(default="collages")
    dpi = luigi.IntParameter(default=1000)

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
            )

    def output(self):
        """"""
        return ValidationLocalTarget(self.collage_base_path)


@copy_params(
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
    collage_base_path=ParamLink(PlotCollage),
    sample=ParamLink(ValidationConfig),
)
class PlotSingleCollage(WorkflowTask):
    """Plot collage for single mtype.

    Args:
        collage_base_path (str): path to the output folder
        mtype (str of list(str)): mtype(s) to plot
        sample (float): number of cells to use, if None, all available
        nb_jobs (int) : number of joblib workers
        joblib_verbose (int) verbose level of joblib
        dpi (int): dpi for pdf rendering (rasterized)
    """

    mtype = luigi.Parameter()
    dpi = luigi.IntParameter(default=1000)

    def requires(self):
        """"""
        return {
            "synthesis": Synthesize(),
            "planes": CreateAtlasPlanes(),
            "layers": CreateAtlasLayerAnnotations(),
        }

    def run(self):
        """"""
        circuit = load_circuit(
            path_to_mvd3=self.input()["synthesis"]["out_mvd3"].path,
            path_to_morphologies=self.input()["synthesis"]["out_morphologies"].path,
            path_to_atlas=CircuitConfig().atlas_path,
        )

        planes = load_planes_centerline(self.input()["planes"].path)["planes"]
        layer_annotation = VoxelData.load_nrrd(
            self.input()["layers"]["annotations"].path
        )
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

    Args:
        scales_base_path (str): path to the output folder
        log_files (str): (optional) directory containing log files to parse
        mtypes (list(str)): mtypes to plot
        neuron_type_position_regex (str): regex used to find neuron type and position
        default_scale_regex (str): regex used to find default scales
        target_scale_regex (str): regex used to find target scales
        neurite_hard_limit_regex (str): regex used to find neurite hard limits
    """

    scales_base_path = luigi.Parameter(default="scales")
    log_files = luigi.OptionalParameter(default=None)
    neuron_type_position_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Neurite type and position: (.*)"
    )
    default_scale_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Default barcode scale: (.*)"
    )
    target_scale_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Target barcode scale: (.*)"
    )
    neurite_hard_limit_regex = luigi.Parameter(
        default=r".*\[WORKER TASK ID=([0-9]*)\] Neurite hard limit rescaling: (.*)"
    )

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
    """Plot collage for single mtype.

    Args:
        output_path (str): path to the output file
        mtypes (list(str)): mtypes to plot
        morphology_path (str): column name to use in the DF from ApplySubstitutionRules
        outlier_percentage (int): percentage from which the outliers are removed
        nb_jobs (int): number of jobs
    """

    output_path = luigi.Parameter(default="path_distance_fit.pdf")
    outlier_percentage = luigi.IntParameter(default=90)

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
    """Create MorphVal reports.

    Args:
        output_path (str): path to the output file
        mtypes (list(str)): mtypes to plot
        morphology_path (str): column name to use in the DF from ApplySubstitutionRules
        outlier_percentage (int): percentage from which the outliers are removed
        nb_jobs (int): number of jobs
    """

    config_path = luigi.OptionalParameter(default=None)
    output_path = luigi.Parameter(default="morphology_validation_reports")
    cell_figure_count = luigi.IntParameter(
        default=10, description="Number of example cells to show"
    )
    bio_compare = BoolParameter(
        default=False, description="Use the bio compare template"
    )

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
