"""Luigi tasks for validation workflows."""
import luigi
import pandas as pd

from synthesis_workflow.tasks.config import ValidationLocalTarget
from synthesis_workflow.tasks.luigi_tools import BoolParameter
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.luigi_tools import WorkflowWrapperTask
from synthesis_workflow.tasks.synthesis import ApplySubstitutionRules
from synthesis_workflow.tasks.utils import GetSynthesisInputs
from synthesis_workflow.tasks.vacuum_synthesis import PlotVacuumMorphologies
from synthesis_workflow.tasks.validation import MorphologyValidationReports
from synthesis_workflow.tasks.validation import PlotCollage
from synthesis_workflow.tasks.validation import PlotDensityProfiles
from synthesis_workflow.tasks.validation import PlotMorphometrics
from synthesis_workflow.tasks.validation import PlotPathDistanceFits
from synthesis_workflow.tasks.validation import PlotScales
from synthesis_workflow.validation import plot_morphometrics


class ValidateSynthesis(WorkflowWrapperTask):
    """Workflow to validate synthesis.

    The complete workflow is described here:

    .. graphviz:: ValidateSynthesis.dot
    """

    with_collage = BoolParameter(default=True, description="Trigger collage.")
    """bool: Trigger collage."""

    with_morphometrics = BoolParameter(
        default=True, description="Trigger morphometrics."
    )
    """bool: Trigger morphometrics."""

    with_density_profiles = BoolParameter(
        default=True, description="Trigger density profiles."
    )
    """bool: Trigger density profiles."""

    with_path_distance_fits = BoolParameter(
        default=True, description="Trigger path distance fits."
    )
    """bool: Trigger path distance fits."""

    with_scale_statistics = BoolParameter(
        default=True, description="Trigger scale statistics."
    )
    """bool: Trigger scale statistics."""

    with_morphology_validation_reports = BoolParameter(
        default=True, description="Trigger morphology validation reports."
    )
    """bool: Trigger morphology validation reports."""

    def requires(self):
        """"""
        tasks = [GetSynthesisInputs()]
        if self.with_collage:
            tasks.append(PlotCollage())
        if self.with_morphometrics:
            tasks.append(PlotMorphometrics(in_atlas=True))
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles(in_atlas=True))
        if self.with_path_distance_fits:
            tasks.append(PlotPathDistanceFits())
        if self.with_scale_statistics:
            tasks.append(PlotScales())
        if self.with_morphology_validation_reports:
            tasks.append(MorphologyValidationReports())
        return tasks


class ValidateVacuumSynthesis(WorkflowWrapperTask):
    """Workflow to validate vacuum synthesis.

    The complete workflow is described here:

    .. graphviz:: ValidateVacuumSynthesis.dot
    """

    with_vacuum_morphologies = BoolParameter(
        default=True, description="Trigger morphologies."
    )
    """bool: Trigger morphologies."""

    with_morphometrics = BoolParameter(
        default=True, description="Trigger morphometrics."
    )
    """bool: Trigger morphometrics."""

    with_density_profiles = BoolParameter(
        default=True, description="Trigger density profiles."
    )
    """bool: Trigger density profiles."""

    def requires(self):
        """"""
        tasks = [GetSynthesisInputs()]
        if self.with_morphometrics:
            tasks.append(PlotMorphometrics(in_atlas=False))
        if self.with_vacuum_morphologies:
            tasks.append(PlotVacuumMorphologies())
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles(in_atlas=False))
        return tasks


class ValidateRescaling(WorkflowTask):
    """Workflow to validate rescaling.

    The complete workflow is described here:

    .. graphviz:: ValidateRescaling.dot
    """

    morphometrics_path = luigi.Parameter(
        default="morphometrics", description="Output path."
    )
    """str: Output path."""

    base_key = luigi.Parameter(
        default="morphology_path", description="Column name in the DF."
    )
    """str: Column name in the DF."""

    comp_key = luigi.Parameter(
        default="morphology_path", description="Column name in the DF."
    )
    """str: Column name in the DF."""

    base_label = luigi.Parameter(
        default="bio", description="Label for the base morphologies."
    )
    """str: Label for the base morphologies."""

    comp_label = luigi.Parameter(
        default="substituted", description="Label for the compared morphologies."
    )
    """str: Label for the compared morphologies."""

    config_features = luigi.DictParameter(
        default=None, description="Mapping of features to plot."
    )
    """dict: Mapping of features to plot."""

    normalize = BoolParameter(description="Normalize data if set to True.")
    """bool: Normalize data if set to True."""

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
        return ValidationLocalTarget(self.morphometrics_path)
