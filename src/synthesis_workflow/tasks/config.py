"""config functions for luigi tasks."""
import logging
import warnings

import luigi

from .luigi_tools import ExtParameter
from .luigi_tools import OptionalIntParameter
from .luigi_tools import OutputLocalTarget


# Add some warning filters
warnings.filterwarnings("ignore", module="diameter_synthesis.build_diameters")
warnings.filterwarnings("ignore", module="joblib")
warnings.filterwarnings("ignore", module="luigi.parameter")
warnings.filterwarnings("ignore", module="neurom.io")
warnings.filterwarnings("ignore", module="neurom.features")
warnings.filterwarnings("ignore", module="scipy")

# Disable matplotlib logger
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.propagate = False


class DiametrizerConfig(luigi.Config):
    """Diametrizer configuration."""

    model = luigi.Parameter(default="generic")
    terminal_threshold = luigi.FloatParameter(default=2.0)
    taper_min = luigi.FloatParameter(default=-0.01)
    taper_max = luigi.FloatParameter(default=1e-6)
    asymmetry_threshold_basal = luigi.FloatParameter(default=1.0)
    asymmetry_threshold_apical = luigi.FloatParameter(default=0.2)
    neurite_types = luigi.ListParameter(default=["basal", "apical"])

    trunk_max_tries = luigi.IntParameter(default=100)
    n_samples = luigi.IntParameter(default=2)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.config_model = {
            "models": [self.model],
            "neurite_types": self.neurite_types,
            "terminal_threshold": self.terminal_threshold,
            "taper": {"max": self.taper_max, "min": self.taper_min},
            "asymmetry_threshold": {
                "apical": self.asymmetry_threshold_apical,
                "basal": self.asymmetry_threshold_basal,
            },
        }

        self.config_diametrizer = {
            "models": [self.model],
            "neurite_types": self.neurite_types,
            "asymmetry_threshold": {
                "apical": self.asymmetry_threshold_apical,
                "basal": self.asymmetry_threshold_basal,
            },
            "trunk_max_tries": self.trunk_max_tries,
            "n_samples": self.n_samples,
        }


class RunnerConfig(luigi.Config):
    """Runner global configuration."""

    nb_jobs = luigi.IntParameter(
        default=-1, description="Number of jobs used by parallel tasks"
    )
    joblib_verbose = luigi.NumericalParameter(
        default=0,
        var_type=int,
        min_value=0,
        max_value=50,
        description="Verbosity level used by the joblib library",
    )


class SynthesisConfig(luigi.Config):
    """Synthesis global configuration."""

    tmd_parameters_path = luigi.Parameter(
        default="tns_input/tmd_parameters.json",
        description="The path to the TMD parameters",
    )
    tmd_distributions_path = luigi.Parameter(
        default="tns_input/tmd_distributions.json",
        description="The path to the TMD distributions",
    )
    cortical_thickness = luigi.ListParameter(
        default=[165, 149, 353, 190, 525, 700],
        description="The list of cortical thicknesses",
    )
    mtypes = luigi.ListParameter(
        default=None, description="The list of mtypes to process"
    )


class CircuitConfig(luigi.Config):
    """Circuit configuration."""

    circuit_somata_path = luigi.Parameter(default="circuit_somata.mvd3")
    atlas_path = luigi.Parameter(default=None)


class PathConfig(luigi.Config):
    """Morphology path configuration."""

    mtype_taxonomy_path = luigi.Parameter(
        default="mtype_taxonomy.tsv", description="path to the taxonomy file (TSV)"
    )
    local_synthesis_input_path = luigi.Parameter(default="synthesis_input")

    result_path = luigi.Parameter(default="out")
    atlas_subpath = luigi.Parameter(default="atlas")
    circuit_subpath = luigi.Parameter(default="circuit")
    morphs_df_subpath = luigi.Parameter(default="morphs_df")
    synthesis_subpath = luigi.Parameter(default="synthesis")
    validation_subpath = luigi.Parameter(default="validation")

    ext = ExtParameter(default="asc")
    morphology_path = luigi.Parameter(default="repaired_morphology_path")
    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    substituted_morphs_df_path = luigi.Parameter(default="substituted_morphs_df.csv")
    synth_morphs_df_path = luigi.Parameter(default="synth_morphs_df.csv")
    synth_output_path = luigi.Parameter(default="synthesized_morphologies")

    debug_region_grower_scales_path = luigi.Parameter(
        default="region_grower_scales_logs"
    )


class ValidationConfig(luigi.Config):
    """Validation configuration."""

    sample = OptionalIntParameter(default=None)


class AtlasLocalTarget(OutputLocalTarget):
    """Specific target for atlas targets"""


class CircuitLocalTarget(OutputLocalTarget):
    """Specific target for circuit targets"""


class MorphsDfLocalTarget(OutputLocalTarget):
    """Specific target for morphology dataframe targets"""


class SynthesisLocalTarget(OutputLocalTarget):
    """Specific target for synthesis targets"""


class ValidationLocalTarget(OutputLocalTarget):
    """Specific target for validation targets"""


def reset_default_prefixes():
    """Set default output paths for targets"""
    OutputLocalTarget.set_default_prefix(PathConfig().result_path)
    AtlasLocalTarget.set_default_prefix(
        # pylint: disable=protected-access
        OutputLocalTarget._prefix
        / PathConfig().atlas_subpath
    )
    CircuitLocalTarget.set_default_prefix(
        # pylint: disable=protected-access
        OutputLocalTarget._prefix
        / PathConfig().circuit_subpath
    )
    MorphsDfLocalTarget.set_default_prefix(
        # pylint: disable=protected-access
        OutputLocalTarget._prefix
        / PathConfig().morphs_df_subpath
    )
    SynthesisLocalTarget.set_default_prefix(
        # pylint: disable=protected-access
        OutputLocalTarget._prefix
        / PathConfig().synthesis_subpath
    )
    ValidationLocalTarget.set_default_prefix(
        # pylint: disable=protected-access
        OutputLocalTarget._prefix
        / PathConfig().validation_subpath
    )


reset_default_prefixes()
