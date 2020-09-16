"""Luigi tasks for validation of synthesis."""
from pathlib import Path
import yaml

import luigi
import pandas as pd

from .synthesis_tasks import CreateAtlasPlanes
from .synthesis_tasks import Synthesize
from .synthesis_tasks import VacuumSynthesize
from .synthesis_tasks import RescaleMorphologies
from .synthesis_tasks import GetSynthetisedNeuriteLengths
from .utils_tasks import BaseWrapperTask
from .utils_tasks import circuitconfigs
from .utils_tasks import load_circuit
from .utils_tasks import pathconfigs
from .validation import convert_mvd3_to_morphs_df
from .validation import plot_collage
from .validation import plot_collage_O1
from .validation import plot_density_profiles
from .validation import plot_morphometrics


class ConvertMvd3(luigi.Task):
    """Convert synthesize mvd3 file to morphs_df.csv file.

    Args:
        ext (str): extension for morphology files
    """

    ext = luigi.Parameter(default=".asc")

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

    morphometrics_path = luigi.Parameter(default="morphometrics")

    def requires(self):
        """"""
        return ConvertMvd3()

    def run(self):
        """"""
        synth_morphs_df = pd.read_csv(self.input().path)
        morphs_df = pd.read_csv(pathconfigs().morphs_df_path)
        plot_morphometrics(morphs_df, synth_morphs_df, self.output().path)

    def output(self):
        """"""
        return luigi.LocalTarget(self.morphometrics_path)


class PlotVacuumMorphometrics(luigi.Task):
    """Plot morphometric."""

    morphometrics_path = luigi.Parameter(default="morphometrics")

    def requires(self):
        """"""
        return {
            "VacuumSynthesize": VacuumSynthesize(),
            "RescaleMorphologies": RescaleMorphologies(),
            "GetSynthetisedNeuriteLengths": GetSynthetisedNeuriteLengths(),
        }

    def run(self):
        """"""
        synth_morphs_df = pd.read_csv(self.input()["VacuumSynthesize"].path)
        morphs_df = pd.read_csv(self.input()["RescaleMorphologies"].path)

        with open(self.input()["GetSynthetisedNeuriteLengths"].path) as f:
            percentile_length = yaml.load(f, Loader=yaml.FullLoader)

        plot_morphometrics(
            morphs_df,
            synth_morphs_df,
            self.output().path,
            bio_key="rescaled_morphology_path",
            synth_key="vacuum_morphology_path",
            vbars=percentile_length,
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
            circuit, self.sample, self.region, self.sample_distance, self.output().path
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.density_profiles_path)


class PlotCollage(BaseWrapperTask):
    """Plot collage.

    Args:
        collage_path (str): path to pdf file
        sample (float): number of cells to use, if None, all available
    """

    collage_base_path = luigi.Parameter(default="collages")
    collage_type = luigi.Parameter(default="O1")
    sample = luigi.IntParameter(default=20)
    mtypes = luigi.ListParameter(default=None)

    def requires(self):
        """"""
        if self.mtypes[0] == "all":
            self.mtypes = pd.read_csv(
                pathconfigs().synth_morphs_df_path
            ).mtype.unique()

        tasks = []
        for mtype in self.mtypes:
            collage_path = (Path(self.collage_base_path) / mtype).with_suffix(".pdf")
            tasks.append(
                PlotSingleCollage(
                    mtype=mtype,
                    collage_path=collage_path,
                    sample=self.sample,
                    collage_type=self.collage_type,
                )
            )
        return tasks


class PlotSingleCollage(luigi.Task):
    """Plot collage for single mtype.

    Args:
        collage_path (str): path to pdf file
        sample (float): number of cells to use, if None, all available
    """

    collage_path = luigi.Parameter(default="collage.pdf")
    sample = luigi.IntParameter(default=20)
    collage_type = luigi.Parameter(default="O1")
    mtype = luigi.Parameter()

    def requires(self):
        """"""
        return {"synthesis": Synthesize(), "planes": CreateAtlasPlanes()}

    def run(self):
        """"""

        circuit = load_circuit(
            path_to_mvd3=self.input()["synthesis"].path,
            path_to_morphologies=pathconfigs().synth_output_path,
            path_to_atlas=circuitconfigs().atlas_path,
        )
        import numpy as np
        from voxcell import VoxelData

        if self.collage_type == "O1":
            plot_collage_O1(circuit, self.sample, self.output().path)

        if self.collage_type == "Isocortex":
            planes = np.load(self.input()["planes"].path)["planes"]
            annotation_path = Path(circuitconfigs().atlas_path) / "layers.nrrd"
            layer_annotation = VoxelData.load_nrrd(annotation_path)
            plot_collage(
                circuit,
                planes,
                layer_annotation,
                self.mtype,
                self.output().path,
                self.sample,
            )

    def output(self):
        """"""
        return luigi.LocalTarget(self.collage_path)


class ValidateSynthesis(luigi.WrapperTask):
    """Main class to validate synthesis."""

    def requires(self):
        """"""
        tasks = [PlotMorphometrics(), PlotDensityProfiles(), PlotCollage()]
        # tasks = [PlotMorphometrics(), PlotDensityProfiles()]
        # tasks = [PlotCollage()]
        return tasks
