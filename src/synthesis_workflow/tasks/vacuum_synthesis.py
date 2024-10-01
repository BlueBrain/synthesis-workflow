"""Luigi tasks for morphology synthesis in vacuum."""

import json
import logging

import luigi
import morphio
import pandas as pd
from luigi.parameter import PathParameter
from luigi_tools.task import ParamRef
from luigi_tools.task import WorkflowTask
from luigi_tools.task import copy_params

from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import MorphsDfLocalTarget
from synthesis_workflow.tasks.config import RunnerConfig
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.config import SynthesisLocalTarget
from synthesis_workflow.tasks.config import ValidationLocalTarget
from synthesis_workflow.tasks.synthesis import BuildSynthesisDistributions
from synthesis_workflow.tasks.synthesis import BuildSynthesisParameters
from synthesis_workflow.vacuum_synthesis import VACUUM_SYNTH_MORPHOLOGY_PATH
from synthesis_workflow.vacuum_synthesis import grow_vacuum_morphologies
from synthesis_workflow.vacuum_synthesis import plot_vacuum_morphologies

morphio.set_maximum_warnings(0)

L = logging.getLogger(__name__)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
    nb_jobs=ParamRef(RunnerConfig),
    joblib_verbose=ParamRef(RunnerConfig),
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
    vacuum_synth_morphs_df_path = PathParameter(
        default="vacuum_synth_morphs_df.csv",
        description="Path to the morphs_df.csv file.",
    )
    diametrizer = luigi.ChoiceParameter(
        default="external",
        choices=["external"] + [f"M{i}" for i in range(1, 6)],
        description=":str: Diametrizer model to use.",
    )
    n_cells = luigi.IntParameter(default=10, description=":int: Number of cells to synthesize.")

    def requires(self):
        """Required input tasks."""
        return {
            "tmd_parameters": BuildSynthesisParameters(),
            "tmd_distributions": BuildSynthesisDistributions(),
        }

    def run(self):
        """Actual process of the task."""
        tmd_parameters = json.load(self.input()["tmd_parameters"].open())
        tmd_distributions = json.load(self.input()["tmd_distributions"].open())

        if self.mtypes is None:
            mtypes = list(tmd_parameters[CircuitConfig().region].keys())
        else:
            mtypes = self.mtypes

        morphology_base_path = self.output()["out_morphologies"].pathlib_path
        morphology_base_path.mkdir(parents=True, exist_ok=True)
        vacuum_synth_morphs_df = grow_vacuum_morphologies(
            mtypes,
            self.n_cells,
            tmd_parameters,
            tmd_distributions,
            morphology_base_path.absolute(),
            CircuitConfig().region,
            vacuum_morphology_path=self.vacuum_synth_morphology_path,
            diametrizer=self.diametrizer,
            joblib_verbose=self.joblib_verbose,
            nb_jobs=self.nb_jobs,
        )
        vacuum_synth_morphs_df.to_csv(self.output()["out_morphs_df"].path, index=False)

    def output(self):
        """Outputs of the task."""
        return {
            "out_morphs_df": MorphsDfLocalTarget(self.vacuum_synth_morphs_df_path),
            "out_morphologies": SynthesisLocalTarget(self.vacuum_synth_morphology_path),
        }


@copy_params(
    vacuum_synth_morphology_path=ParamRef(VacuumSynthesize),
)
class PlotVacuumMorphologies(WorkflowTask):
    """Plot morphologies to obtain annotations.

    Attributes:
        vacuum_synth_morphology_path (str): Column name to use from the morphology DataFrame.
    """

    pdf_filename = PathParameter(
        default="vacuum_morphologies.pdf", description=":str: Path to the output file."
    )

    def requires(self):
        """Required input tasks."""
        return VacuumSynthesize()

    def run(self):
        """Actual process of the task."""
        vacuum_synth_morphs_df = pd.read_csv(self.input()["out_morphs_df"].path)
        plot_vacuum_morphologies(
            vacuum_synth_morphs_df,
            self.output().path,
            self.vacuum_synth_morphology_path,
        )

    def output(self):
        """Outputs of the task."""
        return ValidationLocalTarget(self.pdf_filename)
