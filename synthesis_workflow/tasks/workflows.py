"""Luigi tasks for validation workflows."""
import luigi

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
                bio_key="rescaled_morphology_path",
                synth_key="vacuum_morphology_path",
                morph_type="in_vacuum",
            ),
            PlotVacuumMorphologies(),
            PlotDensityProfiles(),
        ]
        return tasks
