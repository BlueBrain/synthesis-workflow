"""Luigi tasks for validation workflows."""
import luigi
import pandas as pd

from ..validation import plot_morphometrics
from .luigi_tools import WorkflowTask
from .luigi_tools import WorkflowWrapperTask
from .synthesis import ApplySubstitutionRules
from .synthesis import BuildCircuit
from .synthesis import BuildMorphsDF
from .validation import PlotCollage
from .validation import PlotDensityProfiles
from .validation import PlotMorphometrics
from .validation import PlotPathDistanceFits
from .validation import PlotScales
from .vacuum_synthesis import PlotVacuumMorphologies


class ValidateSynthesis(WorkflowWrapperTask):
    """Workflow to validate synthesis"""

    build_circuit = luigi.BoolParameter(default=False)
    build_morphs_df = luigi.BoolParameter(default=False)
    with_collage = luigi.BoolParameter(default=True)
    with_morphometrics = luigi.BoolParameter(default=True)
    with_density_profiles = luigi.BoolParameter(default=True)
    with_path_distance_fits = luigi.BoolParameter(default=True)
    with_scale_statistics = luigi.BoolParameter(default=True)

    def requires(self):
        """"""
        tasks = []
        if self.build_circuit:
            tasks.append(BuildCircuit())
        if self.build_morphs_df:
            tasks.append(BuildMorphsDF())
        if self.with_collage:
            tasks.append(PlotCollage())
        if self.with_morphometrics:
            tasks.append(PlotMorphometrics())
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles())
        if self.with_path_distance_fits:
            tasks.append(PlotPathDistanceFits())
        if self.with_scale_statistics:
            tasks.append(PlotScales())
        return tasks


class ValidateVacuumSynthesis(WorkflowWrapperTask):
    """Workflow to validate vacuum synthesis"""

    with_vacuum_morphologies = luigi.BoolParameter(default=True)
    with_morphometrics = luigi.BoolParameter(default=True)
    with_density_profiles = luigi.BoolParameter(default=True)

    def requires(self):
        """"""
        tasks = []
        if self.with_morphometrics:
            tasks.append(
                PlotMorphometrics(
                    base_key="morphology_path",
                    comp_key="vacuum_morphology_path",
                    morph_type="in_vacuum",
                )
            )
        if self.with_vacuum_morphologies:
            tasks.append(PlotVacuumMorphologies())
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles(region="in_vacuum"))
        return tasks


class ValidateRescaling(WorkflowTask):
    """Workflow to validate rescaling"""

    morphometrics_path = luigi.Parameter(default="morphometrics")
    base_key = luigi.Parameter(default="morphology_path")
    comp_key = luigi.Parameter(default="morphology_path")
    base_label = luigi.Parameter(default="bio")
    comp_label = luigi.Parameter(default="substituted")
    config_features = luigi.DictParameter(default=None)
    normalize = luigi.BoolParameter()

    def requires(self):
        """"""
        # pylint: disable=no-self-use
        return ApplySubstitutionRules()

    def run(self):
        """"""
        # TODO: just call the PlotMorphometrics task with correct arguments?
        base_morphs_df = pd.read_csv(self.requires().input().path)
        comp_morphs_df = pd.read_csv(self.input().path)

        plot_morphometrics(
            base_morphs_df,
            comp_morphs_df,
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
