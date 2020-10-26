"""Luigi tasks for validation of synthesis."""
import logging
from pathlib import Path

import luigi
import pandas as pd
from voxcell import VoxelData
from atlas_analysis.planes.planes import load_planes_centerline

from ..tools import load_circuit
from ..validation import convert_mvd3_to_morphs_df
from ..validation import parse_log
from ..validation import plot_collage
from ..validation import plot_density_profiles
from ..validation import plot_morphometrics
from ..validation import plot_path_distance_fits
from ..validation import plot_scale_statistics
from ..validation import VacuumCircuit
from .circuit import CreateAtlasPlanes
from .circuit import CreateAtlasLayerAnnotations
from .config import CircuitConfig
from .config import PathConfig
from .config import RunnerConfig
from .config import SynthesisConfig
from .config import ValidationConfig
from .config import ValidationLocalTarget
from .luigi_tools import BoolParameter
from .luigi_tools import copy_params
from .luigi_tools import OutputLocalTarget
from .luigi_tools import ParamLink
from .luigi_tools import WorkflowTask
from .luigi_tools import WorkflowError
from .synthesis import BuildMorphsDF
from .synthesis import AddScalingRulesToParameters
from .synthesis import BuildSynthesisDistributions
from .synthesis import ApplySubstitutionRules
from .synthesis import Synthesize
from .vacuum_synthesis import VacuumSynthesize


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
        return OutputLocalTarget(PathConfig().synth_morphs_df_path)


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
    comp_key = luigi.Parameter(default="synth_morphology_path")
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
        else:
            synth_morphs_df = pd.read_csv(self.input()["vacuum"]["out_morphs_df"].path)

        plot_morphometrics(
            morphs_df,
            synth_morphs_df,
            self.output().path,
            base_key=self.base_key,
            comp_key=self.comp_key,
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
        if (
            # pylint: disable=unsubscriptable-object
            self.mtypes is None
            or self.mtypes[0] == "all"
        ):
            mtypes = pd.read_csv(PathConfig().synth_morphs_df_path).mtype.unique()
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
        if (
            self.mtypes is None
            or self.mtypes[0] == "all"  # pylint: disable=unsubscriptable-object
        ):
            mtypes = pd.read_csv(PathConfig().synth_morphs_df_path).mtype.unique()
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