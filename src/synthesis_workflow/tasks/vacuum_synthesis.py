"""Luigi tasks for morphology synthesis in vacuum."""
import json
import logging
from pathlib import Path

import luigi
import morphio
import pandas as pd

from synthesis_workflow.tasks.config import MorphsDfLocalTarget
from synthesis_workflow.tasks.config import RunnerConfig
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.config import SynthesisLocalTarget
from synthesis_workflow.tasks.config import ValidationLocalTarget
from synthesis_workflow.tasks.luigi_tools import copy_params
from synthesis_workflow.tasks.luigi_tools import ParamLink
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.synthesis import BuildSynthesisDistributions
from synthesis_workflow.tasks.synthesis import BuildSynthesisParameters
from synthesis_workflow.tools import ensure_dir
from synthesis_workflow.vacuum_synthesis import grow_vacuum_morphologies
from synthesis_workflow.vacuum_synthesis import plot_vacuum_morphologies
from synthesis_workflow.vacuum_synthesis import VACUUM_SYNTH_MORPHOLOGY_PATH


morphio.set_maximum_warnings(0)

L = logging.getLogger(__name__)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
    nb_jobs=ParamLink(RunnerConfig),
    joblib_verbose=ParamLink(RunnerConfig),
)
class VacuumSynthesize(WorkflowTask):
    """Grow cells in vacuum, for annotation tasks.

    Attributes:
        mtypes (list(str)): List of mtypes to plot.
        nb_jobs (int): Number of jobs.
        joblib_verbose (int): Verbosity level of joblib.
    """

    vacuum_synth_morphology_path = luigi.Parameter(
        default=VACUUM_SYNTH_MORPHOLOGY_PATH,
        description="Name of the column in the morphs_df.csv file.",
    )
    vacuum_synth_morphs_df_path = luigi.Parameter(
        default="vacuum_synth_morphs_df.csv",
        description="Path to the morphs_df.csv file.",
    )
    diametrizer = luigi.ChoiceParameter(
        default="external",
        choices=["external"] + [f"M{i}" for i in range(1, 6)],
        description=":str: Diametrizer model to use.",
    )
    n_cells = luigi.IntParameter(
        default=10, description=":int: Number of cells to synthesize."
    )

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

        if self.mtypes is None:
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
            vacuum_morphology_path=self.vacuum_synth_morphology_path,
            diametrizer=self.diametrizer,
            joblib_verbose=self.joblib_verbose,
            nb_jobs=self.nb_jobs,
        )
        vacuum_synth_morphs_df.to_csv(self.output()["out_morphs_df"].path, index=False)

    def output(self):
        """"""
        return {
            "out_morphs_df": MorphsDfLocalTarget(self.vacuum_synth_morphs_df_path),
            "out_morphologies": SynthesisLocalTarget(self.vacuum_synth_morphology_path),
        }


@copy_params(
    vacuum_synth_morphology_path=ParamLink(VacuumSynthesize),
)
class PlotVacuumMorphologies(WorkflowTask):
    """Plot morphologies to obtain annotations.

    The generated images are like the following:

    .. image:: vacuum_morphologies-1.png

    Attributes:
        vacuum_synth_morphology_path (str): Column name to use from the morphlogy DataFrame.
    """

    pdf_filename = luigi.Parameter(
        default="vacuum_morphologies.pdf", description=":str: Path to the output file."
    )

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
            self.vacuum_synth_morphology_path,
        )

    def output(self):
        """"""
        return ValidationLocalTarget(self.pdf_filename)
