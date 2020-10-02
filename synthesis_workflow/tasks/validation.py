"""Luigi tasks for validation of synthesis."""
import logging
from pathlib import Path

import luigi
import pandas as pd
from voxcell import VoxelData
from atlas_analysis.planes.planes import load_planes_centerline

from ..tools import load_circuit
from ..validation import convert_mvd3_to_morphs_df
from ..validation import plot_collage
from ..validation import plot_density_profiles
from ..validation import plot_morphometrics
from .circuit import CreateAtlasPlanes
from .circuit import CreateAtlasLayerAnnotations
from .config import circuitconfigs
from .config import pathconfigs
from .synthesis import RescaleMorphologies
from .synthesis import Synthesize
from .utils import BaseTask
from .utils import ExtParameter
from .vacuum_synthesis import VacuumSynthesize


L = logging.getLogger(__name__)


class ConvertMvd3(luigi.Task):
    """Convert synthesize mvd3 file to morphs_df.csv file.

    Args:
        ext (str): extension for morphology files
    """

    ext = ExtParameter(default="asc")

    def requires(self):
        """"""
        return Synthesize()

    def run(self):
        """"""
        synth_morphs_df = convert_mvd3_to_morphs_df(
            self.input().path, pathconfigs().synth_output_path, self.ext
        )

        synth_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(pathconfigs().synth_morphs_df_path)


class PlotMorphometrics(luigi.Task):
    """Plot morphometric."""

    morph_type = luigi.Parameter(default="in_circuit")
    config_features = luigi.DictParameter(default=None)
    morphometrics_path = luigi.Parameter(default="morphometrics")
    base_key = luigi.Parameter(default="repaired_morphology_path")
    comp_key = luigi.Parameter(default="synth_morphology_path")
    base_label = luigi.Parameter(default="bio")
    comp_label = luigi.Parameter(default="synth")
    normalize = luigi.BoolParameter(
        default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING
    )

    def run(self):
        """"""
        if self.morph_type == "in_vacuum":
            synthesize_task = yield VacuumSynthesize()
            synth_morphs_df = pd.read_csv(synthesize_task.path)

            rescalemorphologies_task = yield RescaleMorphologies()
            morphs_df = pd.read_csv(rescalemorphologies_task.path)

        elif self.morph_type == "in_circuit":
            convertmvd3_task = yield ConvertMvd3()
            synth_morphs_df = pd.read_csv(convertmvd3_task.path)

            morphs_df = pd.read_csv(pathconfigs().morphs_df_path)

        else:
            raise ValueError(
                "The 'morph_type' argument must be in ['in_circuit', 'in_vacuum']"
            )

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


class PlotDensityProfiles(luigi.Task):
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
    nb_jobs = luigi.IntParameter(default=-1)

    def requires(self):
        """"""
        return Synthesize()

    def run(self):
        """"""

        circuit = load_circuit(
            path_to_mvd3=self.input().path,
            path_to_morphologies=pathconfigs().synth_output_path,
            path_to_atlas=circuitconfigs().atlas_path,
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


class PlotCollage(BaseTask):
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
    mtypes = luigi.ListParameter(default=None)
    nb_jobs = luigi.IntParameter(default=-1)
    joblib_verbose = luigi.IntParameter(default=10)
    dpi = luigi.IntParameter(default=1000)

    def requires(self):
        """"""
        return ConvertMvd3()

    def run(self):
        """"""
        if self.mtypes[0] == "all":  # pylint: disable=unsubscriptable-object
            mtypes = pd.read_csv(pathconfigs().synth_morphs_df_path).mtype.unique()
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


class PlotSingleCollage(BaseTask):
    """Plot collage for single mtype.

    Args:
        collage_base_path (str): path to the output folder
        mtype (str of list(str)): mtype(s) to plot
        sample (float): number of cells to use, if None, all available
        nb_jobs (int) : number of joblib workers
        joblib_verbose (int) verbose level of joblib
        dpi (int): dpi for pdf rendering (rasterized)
    """

    collage_base_path = luigi.Parameter(default="collages")
    mtype = luigi.Parameter()
    sample = luigi.IntParameter(default=20)
    nb_jobs = luigi.IntParameter(default=-1)
    joblib_verbose = luigi.IntParameter(default=10)
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
            path_to_mvd3=self.input()["synthesis"].path,
            path_to_morphologies=pathconfigs().synth_output_path,
            path_to_atlas=circuitconfigs().atlas_path,
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
