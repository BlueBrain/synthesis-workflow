"""Luigi tasks for validation workflows."""
import luigi
import pandas as pd

from ..validation import plot_morphometrics
from .synthesis import RescaleMorphologies
from .validation import PlotCollage
from .validation import PlotDensityProfiles
from .validation import PlotMorphometrics
from .vacuum_synthesis import PlotVacuumMorphologies


class ValidateSynthesis(luigi.WrapperTask):
    """Main class to validate synthesis."""

    def requires(self):
        """"""
        tasks = [PlotMorphometrics(), PlotDensityProfiles(), PlotCollage()]
        return tasks


class ValidateVacuumSynthesis(luigi.WrapperTask):
    """Main class to validate vacuum synthesis."""

    def requires(self):
        """"""
        tasks = [
            PlotMorphometrics(
                base_key="rescaled_morphology_path",
                comp_key="vacuum_morphology_path",
                morph_type="in_vacuum",
            ),
            PlotVacuumMorphologies(),
            PlotDensityProfiles(),
        ]
        return tasks


class ValidateRescaling(luigi.Task):
    """Main class to validate rescaling."""

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
