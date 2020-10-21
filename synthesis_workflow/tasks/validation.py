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
from .circuit import CreateAtlasPlanes
from .circuit import CreateAtlasLayerAnnotations
from .config import CircuitConfig
from .config import PathConfig
from .config import RunnerConfig
from .config import SynthesisConfig
from .luigi_tools import copy_params
from .luigi_tools import ParamLink
from .luigi_tools import WorkflowTask
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
            self.input()["out_mvd3"].path, PathConfig().synth_output_path, self.ext
        )

        synth_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(PathConfig().synth_morphs_df_path)


class PlotMorphometrics(WorkflowTask):
    """Plot morphometric."""

    morph_type = luigi.ChoiceParameter(
        default="in_circuit", choices=["in_circuit", "in_vacuum"]
    )
    config_features = luigi.DictParameter(default=None)
    morphometrics_path = luigi.Parameter(default="morphometrics")
    base_key = luigi.Parameter(default="repaired_morphology_path")
    comp_key = luigi.Parameter(default="synth_morphology_path")
    base_label = luigi.Parameter(default="bio")
    comp_label = luigi.Parameter(default="synth")
    normalize = luigi.BoolParameter()

    @staticmethod
    def _wrong_morph_type():
        raise ValueError(
            "The 'morph_type' argument must be in ['in_circuit', 'in_vacuum']"
        )

    def requires(self):
        """"""
        if self.morph_type == "in_vacuum":
            return [VacuumSynthesize(), ApplySubstitutionRules()]
        elif self.morph_type == "in_circuit":
            return ConvertMvd3()
        else:
            return self._wrong_morph_type()

    def run(self):
        """"""
        if self.morph_type == "in_vacuum":
            synthesize_task = self.input()[0]
            synth_morphs_df = pd.read_csv(synthesize_task.path)

            rescalemorphologies_task = self.input()[1]
            morphs_df = pd.read_csv(rescalemorphologies_task.path)

        elif self.morph_type == "in_circuit":
            convertmvd3_task = self.input()
            synth_morphs_df = pd.read_csv(convertmvd3_task.path)

            morphs_df = pd.read_csv(PathConfig().morphs_df_path)

        else:
            self._wrong_morph_type()

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
        return luigi.LocalTarget(self.morphometrics_path)


@copy_params(
    nb_jobs=ParamLink(RunnerConfig),
)
class PlotDensityProfiles(WorkflowTask):
    """Plot density profiles of neurites in an atlas.

    Args:
        density_profiles_path (str): path for pdf file
        sample_distance (float): distance between sampled points along neurites
        sample (float): number of cells to use, if None, all available
        region (str): name of the region (O1, etc...)
    """

    density_profiles_path = luigi.Parameter(default="density_profiles.pdf")
    sample_distance = luigi.FloatParameter(default=10)
    sample = luigi.IntParameter(default=None)
    region = luigi.Parameter(default="O1")

    def requires(self):
        """"""
        return Synthesize()

    def run(self):
        """"""

        circuit = load_circuit(
            path_to_mvd3=self.input()["out_mvd3"].path,
            path_to_morphologies=PathConfig().synth_output_path,
            path_to_atlas=CircuitConfig().atlas_path,
        )

        plot_density_profiles(
            circuit,
            self.sample,
            self.region,
            self.sample_distance,
            self.output().path,
            self.nb_jobs,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.density_profiles_path)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
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
    sample = luigi.IntParameter(default=20)
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
        return luigi.LocalTarget(self.collage_base_path)


@copy_params(
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
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

    collage_base_path = luigi.Parameter()
    mtype = luigi.Parameter()
    sample = luigi.IntParameter()
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
            path_to_morphologies=PathConfig().synth_output_path,
            path_to_atlas=CircuitConfig().atlas_path,
        )

        planes = load_planes_centerline(self.input()["planes"].path)["planes"]
        layer_annotation = VoxelData.load_nrrd(self.input()["layers"].path)
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
        return luigi.LocalTarget(
            (Path(self.collage_base_path) / self.mtype).with_suffix(".pdf")
        )


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
)
class PlotScales(WorkflowTask):
    """Plot scales.

    Args:
        scales_base_path (str): path to the output folder
        log_file (str): log file to parse
        mtypes (list(str)): mtypes to plot
        neuron_type_position_regex (str): regex used to find neuron type and position
        default_scale_regex (str): regex used to find default scales
        target_scale_regex (str): regex used to find target scales
        neurite_hard_limit_regex (str): regex used to find neurite hard limits
    """

    scales_base_path = luigi.Parameter(default="scales")
    log_file = luigi.Parameter(default="synthesis_workflow.log")
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

        # Plot statistics
        scale_data = parse_log(
            self.log_file,
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
        return luigi.LocalTarget(self.scales_base_path)


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
            SynthesisConfig().tmd_distributions_path,
            self.input()["rescaled"].path,
            self.morphology_path,
            self.output().path,
            self.mtypes,
            self.outlier_percentage,
            self.nb_jobs,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.output_path)
