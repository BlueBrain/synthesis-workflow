"""Configurations for luigi tasks."""

import logging
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import luigi
import yaml
from git import Repo
from luigi.parameter import OptionalChoiceParameter
from luigi_tools.parameter import ExtParameter
from luigi_tools.target import OutputLocalTarget
from luigi_tools.task import WorkflowTask

# Add some warning filters
warnings.filterwarnings("ignore", module="diameter_synthesis.build_diameters")
warnings.filterwarnings("ignore", module="joblib")
warnings.filterwarnings("ignore", module="luigi.parameter")
warnings.filterwarnings("ignore", module="neurom.io")
warnings.filterwarnings("ignore", module="neurom.features")
warnings.filterwarnings("ignore", module="scipy")

# Disable some loggers
logging.getLogger("matplotlib").propagate = False
logging.getLogger("numexpr").propagate = False
logging.getLogger("neurots").propagate = False


class GitClone(WorkflowTask):
    """Task to clone a git repository."""

    url = luigi.Parameter(
        description=(
            ":str: Url of repository. If None, git_synthesis_input_path should be an existing "
            "folder."
        )
    )
    dest = luigi.Parameter(description=":str: Path to the destination directory.")
    branch = luigi.Parameter(default="main")

    def run(self):
        """Actual process of the task."""
        Repo.clone_from(self.url, self.output().path, branch=self.branch)

    def output(self):
        """Outputs of the task."""
        return OutputLocalTarget(self.dest)


class GetSynthesisInputs(WorkflowTask):
    """Task to get synthesis input files from a folder on git repository.

    If no url is provided, this task will copy an existing folder to the target location
    given in the 'local_synthesis_input_path' parameter of the 'PathConfig' task.
    """

    url = luigi.OptionalParameter(
        default=None,
        description=(
            ":str: Url of repository. If None, git_synthesis_input_path should be an "
            "existing folder."
        ),
    )
    version = luigi.OptionalParameter(
        default=None, description=":str: Version of repo to checkout."
    )
    git_synthesis_input_path = luigi.Parameter(
        default="synthesis_input",
        description=":str: Path to folder in git repo with synthesis files.",
    )
    branch = luigi.Parameter(default="main")

    def run(self):
        """Actual process of the task."""
        if self.url is None:
            shutil.copytree(self.git_synthesis_input_path, self.output().path)
        else:
            with TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "tmp_repo"
                # Note: can not be called with yield here because of the TemporaryDirectory
                GitClone(url=self.url, dest=dest, branch=self.branch).run()
                if self.version is not None:
                    r = Repo(dest)
                    r.git.checkout(self.version)
                shutil.copytree(dest / self.git_synthesis_input_path, self.output().path)

    def output(self):
        """Outputs of the task."""
        # TODO: it would probably be better to have a specific target for each file
        return OutputLocalTarget(PathConfig().local_synthesis_input_path)


class DiametrizerConfig(luigi.Config):
    """Diametrizer configuration."""

    model = luigi.ChoiceParameter(default="simpler", choices=["generic", "simpler"])
    terminal_threshold = luigi.FloatParameter(default=2.0)
    taper_min = luigi.FloatParameter(default=-0.01)
    taper_max = luigi.FloatParameter(default=1e-6)
    asymmetry_threshold_basal = luigi.FloatParameter(default=1.0)
    asymmetry_threshold_apical = luigi.FloatParameter(default=0.2)
    neurite_types = luigi.ListParameter(
        default=["basal_dendrite", "apical_dendrite"],
        schema={"type": "array", "items": {"type": "string"}},
    )

    trunk_max_tries = luigi.IntParameter(default=100)
    n_samples = luigi.IntParameter(default=2)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.config_model = {"models": [self.model], "neurite_types": self.neurite_types}
        if self.model == "generic":
            self.config_model.update(
                {
                    "terminal_threshold": self.terminal_threshold,
                    "taper": {"max": self.taper_max, "min": self.taper_min},
                    "asymmetry_threshold": {
                        "apical_dendrite": self.asymmetry_threshold_apical,
                        "basal_dendrite": self.asymmetry_threshold_basal,
                    },
                }
            )

        self.config_diametrizer = {"models": [self.model], "neurite_types": self.neurite_types}
        if self.model == "generic":
            self.config_diametrizer.update(
                {
                    "asymmetry_threshold": {
                        "apical_dendrite": self.asymmetry_threshold_apical,
                        "basal_dendrite": self.asymmetry_threshold_basal,
                    },
                    "trunk_max_tries": self.trunk_max_tries,
                    "n_samples": self.n_samples,
                }
            )


class RunnerConfig(luigi.Config):
    """Runner global configuration."""

    nb_jobs = luigi.IntParameter(
        default=-1, description=":int: Number of jobs used by parallel tasks."
    )
    joblib_verbose = luigi.NumericalParameter(
        default=0,
        var_type=int,
        min_value=0,
        max_value=50,
        description=":int: Verbosity level used by the joblib library.",
    )


class SynthesisConfig(luigi.Config):
    """Synthesis global configuration."""

    tmd_parameters_path = luigi.Parameter(
        default="neurots_input/tmd_parameters.json",
        description=":str: The path to the TMD parameters.",
    )
    tmd_distributions_path = luigi.Parameter(
        default="neurots_input/tmd_distributions.json",
        description=":str: The path to the TMD distributions.",
    )
    mtypes = luigi.OptionalListParameter(
        default=None,
        description=(
            ":list(str): The list of mtypes to process (default is None, which means that all "
            "found mtypes are taken)."
        ),
        schema={"type": "array", "items": {"type": "string"}},
    )
    axon_method = luigi.ChoiceParameter(
        default="no_axon",
        description=":str: The method used to handle axons.",
        choices=["no_axon", "reconstructed", "synthesis"],
    )


class CircuitConfig(luigi.Config):
    """Circuit configuration."""

    circuit_somata_path = luigi.Parameter(
        default="circuit_somata.mvd3", description=":str: Path to the circuit somata."
    )
    atlas_path = luigi.OptionalParameter(
        default=None, description=":str: Path to the atlas directory."
    )
    cell_composition_path = luigi.Parameter(
        default="cell_composition.yaml", description=":str: Path to cell_compoistion.yaml file."
    )
    region_structure_path = luigi.Parameter(
        default="region_structure.yaml",
        description=":str: Path to the file containing the layer structure data.",
    )
    region = luigi.OptionalParameter(default="default")
    hemisphere = OptionalChoiceParameter(
        default=None,
        choices=["left", "right"],
        description=":str: The hemisphere side.",
    )


class GetCellComposition(luigi.Task):
    """Get cell_composition with adapted entries."""

    new_cell_composition = luigi.Parameter(
        default="new_cell_composition.yaml",
        description=":str: Filename to the new cell composition file with modified entries.",
    )

    def requires(self):
        """Required input tasks."""
        return GetSynthesisInputs()

    def run(self):
        """Actual process of the task."""
        if not Path(CircuitConfig().cell_composition_path).exists():
            cell_composition_path = str(
                self.input().pathlib_path / CircuitConfig().cell_composition_path
            )
        else:
            cell_composition_path = CircuitConfig().cell_composition_path

        with open(cell_composition_path, "r", encoding="utf-8") as comp_p:
            cell_comp = yaml.safe_load(comp_p)
        if SynthesisConfig().mtypes is not None:
            new_cell_comp = {"version": cell_comp["version"], "neurons": []}
            for entry in cell_comp["neurons"]:
                if entry["traits"]["mtype"] in SynthesisConfig().mtypes:
                    new_cell_comp["neurons"].append(entry)
        else:
            new_cell_comp = cell_comp
        with open(self.output().path, "w", encoding="utf-8") as comp_p:
            yaml.safe_dump(new_cell_comp, comp_p)

    def output(self):
        """Outputs of the task."""
        return OutputLocalTarget(self.new_cell_composition)


class PathConfig(luigi.Config):
    """Morphology path configuration."""

    # Input paths
    local_synthesis_input_path = luigi.Parameter(
        default="synthesis_input",
        description=":str: Path to the synthesis input directory.",
    )

    # Output tree
    result_path = luigi.Parameter(default="out", description=":str: Path to the output directory.")
    atlas_subpath = luigi.Parameter(
        default="atlas", description=":str: Path to output atlas subdirectory."
    )
    circuit_subpath = luigi.Parameter(
        default="circuit", description=":str: Path to output circuit subdirectory."
    )
    morphs_df_subpath = luigi.Parameter(
        default="morphs_df", description=":str: Path to output morphs_df subdirectory."
    )
    synthesis_subpath = luigi.Parameter(
        default="synthesis", description=":str: Path to output synthesis subdirectory."
    )
    validation_subpath = luigi.Parameter(
        default="validation",
        description=":str: Path to output validation subdirectory.",
    )

    # Default internal values
    ext = ExtParameter(default="asc", description=":str: Default extension used.")
    morphology_path = luigi.Parameter(
        default="path",
        description="Column name in the morphology dataframe to access morphology paths",
    )
    morphs_df_path = luigi.Parameter(
        default="morphs_df.csv", description=":str: Path to the morphology DataFrame."
    )
    substituted_morphs_df_path = luigi.Parameter(
        default="substituted_morphs_df.csv",
        description=":str: Path to the substituted morphology DataFrame.",
    )
    synth_morphs_df_path = luigi.Parameter(
        default="synth_morphs_df.csv",
        description=":str: Path to the synthesized morphology DataFrame.",
    )
    synth_output_path = luigi.Parameter(
        default="synthesized_morphologies",
        description=":str: Path to the synthesized morphologies.",
    )

    debug_region_grower_scales_path = luigi.Parameter(
        default="region_grower_scales.pkl",
        description=(
            ":str: Path to the log files in which the scaling factors computed in region-grower "
            "are stored."
        ),
    )


class AtlasLocalTarget(OutputLocalTarget):
    """Specific target for atlas targets."""


class CircuitLocalTarget(OutputLocalTarget):
    """Specific target for circuit targets."""


class MorphsDfLocalTarget(OutputLocalTarget):
    """Specific target for morphology dataframe targets."""


class SynthesisLocalTarget(OutputLocalTarget):
    """Specific target for synthesis targets."""


class ValidationLocalTarget(OutputLocalTarget):
    """Specific target for validation targets."""


def reset_default_prefixes():
    """Set default output paths for targets."""
    OutputLocalTarget.set_default_prefix(PathConfig().result_path)
    AtlasLocalTarget.set_default_prefix(PathConfig().atlas_subpath)
    CircuitLocalTarget.set_default_prefix(PathConfig().circuit_subpath)
    MorphsDfLocalTarget.set_default_prefix(PathConfig().morphs_df_subpath)
    SynthesisLocalTarget.set_default_prefix(PathConfig().synthesis_subpath)
    ValidationLocalTarget.set_default_prefix(PathConfig().validation_subpath)


reset_default_prefixes()
