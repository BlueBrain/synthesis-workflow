"""Luigi tasks to diametrize cells."""

import logging
import os
import sys
import traceback
from functools import partial
from pathlib import Path

import luigi
import matplotlib
import pandas as pd
import yaml
from diameter_synthesis.build_diameters import build as build_diameters
from diameter_synthesis.build_models import build as build_diameter_model
from diameter_synthesis.plotting import plot_distribution_fit
from joblib import Parallel
from joblib import delayed
from luigi_tools.parameter import BoolParameter
from luigi_tools.task import ParamRef
from luigi_tools.task import WorkflowTask
from luigi_tools.task import copy_params
from morphio.mut import Morphology
from neurom import load_morphologies
from tqdm import tqdm

from synthesis_workflow.tasks.config import DiametrizerConfig
from synthesis_workflow.tasks.config import OutputLocalTarget
from synthesis_workflow.tasks.config import RunnerConfig
from synthesis_workflow.tools import update_morphs_df

matplotlib.use("Agg")

L = logging.getLogger(__name__)


def _build_diameter_model(
    mtype, morphs_df=None, config_model=None, morphology_path="morphology_path"
):
    """Internal model builder for parallelisation."""
    morphologies = load_morphologies(
        morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list()
    )
    return mtype, build_diameter_model(morphologies, config_model, with_data=True)


def _plot_models(models_params, models_data, fig_folder="figures", ext=".png"):
    """Plot the models."""
    L.info("Plot the fits...")

    if not Path(fig_folder).exists():
        os.mkdir(fig_folder)

    for mtype in tqdm(models_params):
        if not (Path(fig_folder) / mtype).exists():
            os.mkdir(Path(fig_folder) / mtype)

        for fit_tpe in models_data[mtype]:
            fig_name = Path(fig_folder) / mtype / fit_tpe
            plot_distribution_fit(
                models_data[mtype][fit_tpe],
                models_params[mtype][fit_tpe],
                list(models_data[mtype][fit_tpe].keys()),
                fig_name=fig_name,
                ext=ext,
            )


@copy_params(
    nb_jobs=ParamRef(RunnerConfig),
)
class BuildDiameterModels(WorkflowTask):
    """Task to build diameter models from set of cells.

    Attributes:
        nb_jobs (int): Number of workers.
    """

    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    morphology_path = luigi.Parameter(default="morphology_path")
    diameter_models_path = luigi.Parameter(default="diameter_models.yaml")
    by_mtypes = BoolParameter()
    plot_models = BoolParameter()

    def run(self):
        """Actual process of the task."""
        raise DeprecationWarning("This task must be updated to be used")
        # pylint: disable=unreachable

        config_model = DiametrizerConfig().config_model
        morphs_df = pd.read_csv(self.morphs_df_path)

        models_params = {}
        models_data = {}
        if self.by_mtypes:
            mtypes = list(set(morphs_df.mtype))

            build_model = partial(
                _build_diameter_model,
                morphs_df=morphs_df,
                config_model=config_model,
                morphology_path=self.morphology_path,
            )
            for mtype, (params, data) in Parallel(self.nb_jobs)(
                delayed(build_model)(mtype) for mtype in tqdm(mtypes)
            ):
                models_params[mtype] = params
                models_data[mtype] = data
        else:
            morphologies = load_morphologies(morphs_df[self.morphology_path].to_list())
            models_params["all"], models_data["all"] = build_model(
                morphologies, config_model, with_data=True
            )

        with self.output().open("w") as f:
            yaml.dump(models_params, f)

        if self.plot_models:
            _plot_models(models_params, models_data, fig_folder="figures", ext=".png")

    def output(self):
        """Outputs of the task."""
        return OutputLocalTarget(self.diameter_models_path)


def _diametrizer(
    gid,
    morphs_df=None,
    models_params=None,
    morphology_path="morphology_path",
    config=None,
    new_morphology_path=None,
):
    try:
        if "all" in models_params:
            model_params = models_params["all"]
        else:
            model_params = models_params[morphs_df.loc[gid, "mtype"]]

        neuron = Morphology(morphs_df.loc[gid, morphology_path])
        build_diameters(neuron, model_params, config["neurite_types"], config)

        new_path = (Path(new_morphology_path).absolute() / morphs_df.loc[gid, "name"]).with_suffix(
            ".asc"
        )
        neuron.write(new_path)
        exception = None
    except Exception:  # pylint: disable=broad-except
        exception = "".join(traceback.format_exception(*sys.exc_info()))
        new_path = None
        L.debug(exception)
    return gid, new_path, exception


class Diametrize(WorkflowTask):
    """Task to build diameter models from set of cells."""

    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    morphology_path = luigi.Parameter(default="morphology_path")
    diameter_models_path = luigi.Parameter(default="diameter_models.yaml")
    new_morphology_path = luigi.Parameter(default="diametrized_morphologies")
    new_morphs_df_path = luigi.Parameter(default="diametrized_morphs_df.csv")

    def requires(self):
        """Required input tasks."""
        return BuildDiameterModels()

    def run(self):
        """Actual process of the task."""
        raise DeprecationWarning("This task must be updated to be used")
        # pylint: disable=unreachable

        config = DiametrizerConfig().config_diametrizer

        with self.input().open() as f:
            models_params = yaml.safe_load(f)

        morphs_df = pd.read_csv(self.morphs_df_path)

        diametrizer = partial(
            _diametrizer,
            morphs_df=morphs_df,
            models_params=models_params,
            morphology_path=self.morphology_path,
            config=config,
            new_morphology_path=self.new_morphology_path,
        )

        exception_count = 0
        for gid, new_path, exception in Parallel(-1)(
            delayed(diametrizer)(gid) for gid in tqdm(morphs_df.index)
        ):
            morphs_df.loc[gid, self.new_morphology_path] = new_path
            morphs_df.loc[gid, "exception"] = exception
            if exception is not None:
                exception_count += 1
        L.info("Diametrization terminated, with %s exceptions.", exception_count)

        update_morphs_df(self.morphs_df_path, morphs_df).to_csv(self.output().path, index=False)

    def output(self):
        """Outputs of the task."""
        return OutputLocalTarget(self.new_morphs_df_path)
