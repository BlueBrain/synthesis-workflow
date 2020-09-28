"""Luigi tasks for validation workflows."""
import luigi
import pandas as pd

from ..validation import plot_morphometrics
from .synthesis import RescaleMorphologies
from .validation import PlotCollage
from .validation import PlotDensityProfiles
from .validation import PlotMorphometrics
from .validation import PlotPathDistanceFits
from .vacuum_synthesis import PlotVacuumMorphologies


class ValidateSynthesis(luigi.WrapperTask):
    """Workflow to validate synthesis"""

    with_collage = luigi.BoolParameter(default=True)
    with_morphometrics = luigi.BoolParameter(default=True)
    with_density_profiles = luigi.BoolParameter(default=True)
    with_path_distance_fits = luigi.BoolParameter(default=True)

    def requires(self):
        """"""
        tasks = []
        if self.with_collage:
            tasks.append(PlotCollage())
        if self.with_morphometrics:
            tasks.append(PlotMorphometrics())
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles())
        if self.with_path_distance_fits:
            tasks.append(PlotPathDistanceFits())
        return tasks


class ValidateVacuumSynthesis(luigi.WrapperTask):
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
                    base_key="rescaled_morphology_path",
                    comp_key="vacuum_morphology_path",
                    morph_type="in_vacuum",
                )
            )
        if self.with_vacuum_morphologies:
            tasks.append(PlotVacuumMorphologies())
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles())
        return tasks


class ValidateRescaling(luigi.Task):
    """Workflow to validate rescaling"""

    morphometrics_path = luigi.Parameter(default="morphometrics")
    base_key = luigi.Parameter(default="morphology_path")
    comp_key = luigi.Parameter(default="rescaled_morphology_path")
    base_label = luigi.Parameter(default="bio")
    comp_label = luigi.Parameter(default="rescaled")
    config_features = luigi.DictParameter(default=None)
    normalize = luigi.BoolParameter(
        default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING
    )

    def requires(self):
        """"""
        return RescaleMorphologies()

    def run(self):
        """"""

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
