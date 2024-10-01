"""Luigi tasks for validation workflows."""

import configparser
import json
import pickle

import luigi
import pandas as pd
from luigi.parameter import PathParameter
from luigi_tools.parameter import BoolParameter
from luigi_tools.target import OutputLocalTarget
from luigi_tools.task import WorkflowTask
from luigi_tools.task import WorkflowWrapperTask

from synthesis_workflow.tasks.circuit import CreateAtlasPlanes
from synthesis_workflow.tasks.circuit import SliceCircuit
from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import GetSynthesisInputs
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.config import ValidationLocalTarget
from synthesis_workflow.tasks.synthesis import ApplySubstitutionRules
from synthesis_workflow.tasks.synthesis import Synthesize
from synthesis_workflow.tasks.vacuum_synthesis import PlotVacuumMorphologies
from synthesis_workflow.tasks.validation import MorphologyValidationReports
from synthesis_workflow.tasks.validation import PlotCollage
from synthesis_workflow.tasks.validation import PlotDensityProfiles
from synthesis_workflow.tasks.validation import PlotMorphometrics
from synthesis_workflow.tasks.validation import PlotPathDistanceFits
from synthesis_workflow.tasks.validation import PlotScales
from synthesis_workflow.tasks.validation import PlotScoreMatrix
from synthesis_workflow.tasks.validation import TrunkValidation
from synthesis_workflow.utils import create_circuit_config
from synthesis_workflow.utils import save_planes
from synthesis_workflow.validation import plot_morphometrics


class CreateCircuitConfig(WorkflowTask):
    """Create a CircuitConfig file to be read with other BBP tools (bluepy, etc.)."""

    circuitconfig_path = PathParameter(default="circuit_config.json")
    collageconfig_path = PathParameter(default="collage_config.ini")

    def requires(self):
        """Required input tasks."""
        return {
            "synthesis": Synthesize(),
            "planes": CreateAtlasPlanes(),
            "synthesis_input": GetSynthesisInputs(),
        }

    def run(self):
        """Actual process of the task."""
        # Convert plane data for collage
        with open(self.input()["planes"].path, "rb") as f_planes:
            planes = pickle.load(f_planes)
        save_planes(planes, self.input()["planes"].pathlib_path.parent)

        config = configparser.ConfigParser()
        config["atlas"] = {
            "path": CircuitConfig().atlas_path,
            "structure_path": self.input()["synthesis_input"].pathlib_path
            / CircuitConfig().region_structure_path,
        }
        config["circuit"] = {"path": self.output().path}
        if CircuitConfig().region is not None:
            config["circuit"]["region"] = CircuitConfig().region
        if CircuitConfig().hemisphere is not None:
            config["circuit"]["hemisphere"] = CircuitConfig().hemisphere

        sample = PlotCollage().sample
        if sample is None:
            sample = SliceCircuit().n_cells
        config["cells"] = {
            "mtypes": (
                json.dumps(list(SynthesisConfig().mtypes))
                if SynthesisConfig().mtypes is not None
                else ""
            ),
            "sample": str(sample),
        }
        config["planes"] = {
            "count": CreateAtlasPlanes().plane_count,
            "type": CreateAtlasPlanes().plane_type,
            "slice_thickness": CreateAtlasPlanes().slice_thickness,
        }
        config["collage"] = {"pdf_filename": "collage.pdf"}
        with open(
            self.output().pathlib_path.parent / self.collageconfig_path, "w", encoding="utf-8"
        ) as configfile:
            config.write(configfile)

        config_dict = create_circuit_config(
            self.input()["synthesis"]["circuit"].pathlib_path.resolve(),
            self.input()["synthesis"]["out_morphologies"].pathlib_path.resolve(),
        )

        with open(self.output().path, "w", encoding="utf-8") as config_file:
            json.dump(config_dict, config_file, indent=2)

    def output(self):
        """Outputs of the task."""
        return OutputLocalTarget(self.circuitconfig_path)


class ValidateSynthesis(WorkflowWrapperTask):
    """Workflow to validate synthesis.

    The complete workflow has the following dependency graph:

    .. graphviz:: ValidateSynthesis.dot
    """

    with_collage = BoolParameter(default=True, description=":bool: Trigger collage.")
    with_morphometrics = BoolParameter(default=True, description=":bool: Trigger morphometrics.")
    with_density_profiles = BoolParameter(
        default=True, description=":bool: Trigger density profiles."
    )
    with_path_distance_fits = BoolParameter(
        default=True, description=":bool: Trigger path distance fits."
    )
    with_scale_statistics = BoolParameter(
        default=True, description=":bool: Trigger scale statistics."
    )
    with_morphology_validation_reports = BoolParameter(
        default=True, description=":bool: Trigger morphology validation reports."
    )
    with_score_matrix_reports = BoolParameter(
        default=True, description=":bool: Trigger score matrix reports."
    )
    with_trunk_validation = BoolParameter(
        default=False, description=":bool: Trigger trunk validation."
    )

    def requires(self):
        """Required input tasks."""
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
        if self.with_score_matrix_reports:
            tasks.append(PlotScoreMatrix(in_atlas=True))
        if self.with_trunk_validation:
            tasks.append(TrunkValidation(in_atlas=True))
        tasks.append(CreateCircuitConfig())
        return tasks


class ValidateVacuumSynthesis(WorkflowWrapperTask):
    """Workflow to validate vacuum synthesis.

    The complete workflow has the following dependency graph:

    .. graphviz:: ValidateVacuumSynthesis.dot
    """

    with_vacuum_morphologies = BoolParameter(
        default=True, description=":bool: Trigger morphologies."
    )
    with_morphometrics = BoolParameter(default=True, description=":bool: Trigger morphometrics.")
    with_density_profiles = BoolParameter(
        default=True, description=":bool: Trigger density profiles."
    )
    with_score_matrix_reports = BoolParameter(
        default=True, description=":bool: Trigger score matrix reports."
    )
    with_trunk_validation = BoolParameter(
        default=False, description=":bool: Trigger trunk validation."
    )

    def requires(self):
        """Required input tasks."""
        tasks = [GetSynthesisInputs()]
        if self.with_morphometrics:
            tasks.append(PlotMorphometrics(in_atlas=False))
        if self.with_vacuum_morphologies:
            tasks.append(PlotVacuumMorphologies())
        if self.with_density_profiles:
            tasks.append(PlotDensityProfiles(in_atlas=False))
        if self.with_score_matrix_reports:
            tasks.append(PlotScoreMatrix(in_atlas=False))
        if self.with_trunk_validation:
            tasks.append(TrunkValidation(in_atlas=False))
        return tasks


class ValidateRescaling(WorkflowTask):
    """Workflow to validate rescaling.

    The complete workflow has the following dependency graph:

    .. graphviz:: ValidateRescaling.dot
    """

    morphometrics_path = luigi.Parameter(default="morphometrics", description=":str: Output path.")
    base_key = luigi.Parameter(
        default="morphology_path", description=":str: Column name in the DF."
    )
    comp_key = luigi.Parameter(
        default="morphology_path", description=":str: Column name in the DF."
    )
    base_label = luigi.Parameter(
        default="bio", description=":str: Label for the base morphologies."
    )
    comp_label = luigi.Parameter(
        default="substituted", description=":str: Label for the compared morphologies."
    )
    config_features = luigi.OptionalDictParameter(
        default=None, description=":dict: Mapping of features to plot."
    )
    normalize = BoolParameter(description=":bool: Normalize data if set to True.")

    def requires(self):
        """Required input tasks."""
        return ApplySubstitutionRules()

    def run(self):
        """Actual process of the task."""
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
        """Outputs of the task."""
        return ValidationLocalTarget(self.morphometrics_path)
