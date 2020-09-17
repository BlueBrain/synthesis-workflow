"""config functions for luigi tasks."""
import logging
import warnings

import luigi


logger = logging.getLogger("luigi-interface")
warnings.filterwarnings("ignore", module="diameter_synthesis.build_diameters")
warnings.filterwarnings("ignore", module="joblib")
warnings.filterwarnings("ignore", module="luigi.parameter")
warnings.filterwarnings("ignore", module="neurom.io")
warnings.filterwarnings("ignore", module="neurom.features")
warnings.filterwarnings("ignore", module="scipy")


class diametrizerconfigs(luigi.Config):
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


class synthesisconfigs(luigi.Config):
    """Circuit configuration."""

    tmd_parameters_path = luigi.Parameter(default="tmd_parameters.json")
    tmd_distributions_path = luigi.Parameter(default="tmd_distributions.json")
    pc_in_types_path = luigi.Parameter(default="pc_in_types.yaml")
    cortical_thickness = luigi.Parameter(default="[165, 149, 353, 190, 525, 700]")
    to_use_flag = luigi.Parameter(default="all")
    mtypes = luigi.ListParameter(default=["all"])


class circuitconfigs(luigi.Config):
    """Circuit configuration."""

    circuit_somata_path = luigi.Parameter(default="circuit_somata.mvd3")
    atlas_path = luigi.Parameter(default=None)


class pathconfigs(luigi.Config):
    """Morphology path configuration."""

    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    morphology_path = luigi.Parameter(default="morphology_path")
    synth_morphs_df_path = luigi.Parameter(default="synth_morphs_df.csv")
    synth_output_path = luigi.Parameter(default="synthesized_morphologies")
    substituted_morphs_df_path = luigi.Parameter(default="substituted_morphs_df.csv")
