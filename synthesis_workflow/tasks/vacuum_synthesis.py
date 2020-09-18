"""Luigi tasks for morphology synthesis."""
import json
from pathlib import Path

import luigi
import morphio
import pandas as pd
import yaml

from ..synthesis import get_mean_neurite_lengths
from ..tools import ensure_dir
from ..vacuum_synthesis import grow_vacuum_morphologies
from ..vacuum_synthesis import plot_vacuum_morphologies
from .config import logger as L
from .synthesis import BuildSynthesisDistributions
from .synthesis import BuildSynthesisParameters
from .synthesis import GetMeanNeuriteLengths
from .synthesis import RescaleMorphologies
from .utils import BaseTask


morphio.set_maximum_warnings(0)


class VacuumSynthesize(BaseTask):
    """Grow cells in vacuum, for annotation tasks."""

    mtypes = luigi.ListParameter(default=None)
    vacuum_synth_morphology_path = luigi.Parameter(default="vacuum_synth_morphologies")
    vacuum_synth_morphs_df_path = luigi.Parameter(default="vacuum_synth_morphs_df.csv")
    n_cells = luigi.IntParameter(default=10)
    nb_jobs = luigi.IntParameter(default=-1)
    joblib_verbose = luigi.IntParameter(default=10)

    def requires(self):
        """"""

        return {
            "tmd_parameters": BuildSynthesisParameters(),
            "tmd_distributions": BuildSynthesisDistributions(),
        }

    def run(self):
        """"""
        tmd_parameters = json.load(self.input()["tmd_parameters"].open())
        tmd_distributions = json.load(self.input()["tmd_distributions"].open())

        if self.mtypes[0] == "all":  # pylint: disable=unsubscriptable-object
            mtypes = list(tmd_parameters.keys())
        else:
            mtypes = self.mtypes

        Path(self.vacuum_synth_morphology_path).mkdir(parents=True, exist_ok=True)
        morphology_base_path = Path(self.vacuum_synth_morphology_path).absolute()
        vacuum_synth_morphs_df = grow_vacuum_morphologies(
            mtypes,
            self.n_cells,
            tmd_parameters,
            tmd_distributions,
            morphology_base_path,
            joblib_verbose=self.joblib_verbose,
            nb_jobs=self.nb_jobs,
        )
        vacuum_synth_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return luigi.LocalTarget(self.vacuum_synth_morphs_df_path)


class GetSynthetisedNeuriteLengths(BaseTask):
    """Get the mean neurite lenghts of a neuron population, per mtype and neurite type."""

    neurite_types = luigi.ListParameter(default=["apical"])
    mtypes = luigi.ListParameter(default=["all"])
    morphology_path = luigi.Parameter(default="morphology_path")
    mean_lengths_path = luigi.Parameter(default="mean_neurite_lengths.yaml")
    percentile = luigi.Parameter(default=50)

    def requires(self):
        """"""
        return VacuumSynthesize()

    def run(self):
        """"""
        synth_morphs_df = pd.read_csv(self.input().path)
        mean_lengths = {
            neurite_type: get_mean_neurite_lengths(
                synth_morphs_df,
                neurite_type=neurite_type,
                mtypes=self.mtypes,
                morphology_path=self.morphology_path,
                percentile=self.percentile,
            )
            for neurite_type in self.neurite_types  # pylint: disable=not-an-iterable
        }

        L.info("Lengths: %s", mean_lengths)

        with self.output().open("w") as f:
            yaml.dump(mean_lengths, f)

    def output(self):
        """"""
        return luigi.LocalTarget(self.mean_lengths_path)


class PlotVacuumMorphologies(BaseTask):
    """Plot morphologies to obtain annotations."""

    pdf_filename = luigi.Parameter(default="vacuum_morphologies.pdf")
    morphology_path = luigi.Parameter(default="vacuum_morphology_path")

    def requires(self):
        """"""
        return {
            "vacuum": VacuumSynthesize(),
            "mean_lengths": GetMeanNeuriteLengths(),
            "rescaled": RescaleMorphologies(),
        }

    def run(self):
        """"""
        vacuum_synth_morphs_df = pd.read_csv(self.input()["vacuum"].path)
        mean_lengths = yaml.full_load(self.input()["mean_lengths"].open())
        ensure_dir(self.output().path)
        plot_vacuum_morphologies(
            vacuum_synth_morphs_df,
            self.output().path,
            self.morphology_path,
            mean_lengths,
        )

        rescaled_morphs_df = pd.read_csv(self.input()["rescaled"].path)
        plot_vacuum_morphologies(
            rescaled_morphs_df,
            "figures/rescaled.pdf",
            "rescaled_morphology_path",
            mean_lengths,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.pdf_filename)
