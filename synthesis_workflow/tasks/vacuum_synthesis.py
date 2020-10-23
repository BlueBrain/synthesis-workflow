"""Luigi tasks for morphology synthesis."""
import json
import logging
from pathlib import Path

import luigi
import morphio
import pandas as pd

from ..tools import ensure_dir
from ..vacuum_synthesis import grow_vacuum_morphologies
from ..vacuum_synthesis import plot_vacuum_morphologies
from .config import PathConfig
from .config import RunnerConfig
from .config import SynthesisConfig
from .luigi_tools import copy_params
from .luigi_tools import ParamLink
from .luigi_tools import WorkflowTask
from .synthesis import BuildSynthesisDistributions
from .synthesis import BuildSynthesisParameters


morphio.set_maximum_warnings(0)

L = logging.getLogger(__name__)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
)
class VacuumSynthesize(WorkflowTask):
    """Grow cells in vacuum, for annotation tasks."""

    vacuum_synth_morphology_path = luigi.Parameter(default="vacuum_synth_morphologies")
    vacuum_synth_morphs_df_path = luigi.Parameter(default="vacuum_synth_morphs_df.csv")
    diametrizer = luigi.ChoiceParameter(
        default="external", choices=["external"] + [f"M{i}" for i in range(1, 6)]
    )
    n_cells = luigi.IntParameter(default=10)

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

        if (
            # pylint: disable=unsubscriptable-object
            self.mtypes is None
            or self.mtypes[0] == "all"
        ):
            mtypes = list(tmd_parameters.keys())
        else:
            mtypes = self.mtypes

        Path(self.output()["out_morphologies"].path).mkdir(parents=True, exist_ok=True)
        morphology_base_path = Path(self.output()["out_morphologies"].path).absolute()
        vacuum_synth_morphs_df = grow_vacuum_morphologies(
            mtypes,
            self.n_cells,
            tmd_parameters,
            tmd_distributions,
            morphology_base_path,
            diametrizer=self.diametrizer,
            joblib_verbose=self.joblib_verbose,
            nb_jobs=self.nb_jobs,
        )
        vacuum_synth_morphs_df.to_csv(self.output()["out_morphs_df"].path, index=False)

    def output(self):
        """"""
        return {
            "out_morphs_df": luigi.LocalTarget(self.vacuum_synth_morphs_df_path),
            "out_morphologies": luigi.LocalTarget(self.vacuum_synth_morphology_path),
        }


@copy_params(
    morphology_path=ParamLink(PathConfig),
)
class PlotVacuumMorphologies(WorkflowTask):
    """Plot morphologies to obtain annotations."""

    pdf_filename = luigi.Parameter(default="vacuum_morphologies.pdf")

    def requires(self):
        """"""
        return VacuumSynthesize()

    def run(self):
        """"""
        vacuum_synth_morphs_df = pd.read_csv(self.input()["out_morphs_df"].path)
        ensure_dir(self.output().path)
        plot_vacuum_morphologies(
            vacuum_synth_morphs_df,
            self.output().path,
            self.morphology_path,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.pdf_filename)
