"""Functions for synthesis in vacuum to be used by luigi tasks."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diameter_synthesis import build_diameters
from joblib import Parallel
from joblib import cpu_count
from joblib import delayed
from matplotlib.backends.backend_pdf import PdfPages
from neurom import load_morphology
from neurom.view import matplotlib_impl
from neurots import NeuronGrower
from tqdm import tqdm

from synthesis_workflow.utils import DisableLogger

VACUUM_SYNTH_MORPHOLOGY_PATH = "vacuum_synth_morphologies"


def _grow_morphology(
    gid,
    mtype,
    tmd_parameters,
    tmd_distributions,
    morphology_base_path,
    vacuum_morphology_path=VACUUM_SYNTH_MORPHOLOGY_PATH,
    external_diametrizer=None,
):
    """Grow single morphology for parallel computations."""
    name = f"vacuum_{gid}.asc"
    morphology_path = morphology_base_path / name
    vacuum_synth_morphs_df = pd.DataFrame()
    np.random.seed(gid)

    grower = NeuronGrower(
        input_parameters=tmd_parameters,
        input_distributions=tmd_distributions,
        external_diametrizer=external_diametrizer,
    )
    grower.grow()
    grower.neuron.write(morphology_path)

    vacuum_synth_morphs_df.loc[gid, "name"] = name
    vacuum_synth_morphs_df.loc[gid, "mtype"] = mtype
    vacuum_synth_morphs_df.loc[gid, vacuum_morphology_path] = morphology_path
    # vacuum_synth_morphs_df.loc[gid, 'apical_point'] = grower.apical_points

    return vacuum_synth_morphs_df


def _external_diametrizer(neuron, neurite_type, model_all, random_generator, diameter_params=None):
    return build_diameters.build(
        neuron,
        model_all,
        [neurite_type],
        diameter_params,
        random_generator,
    )


def grow_vacuum_morphologies(
    mtypes,
    n_cells,
    tmd_parameters,
    tmd_distributions,
    morphology_base_path,
    vacuum_morphology_path=VACUUM_SYNTH_MORPHOLOGY_PATH,
    diametrizer="external",
    joblib_verbose=0,
    nb_jobs=1,
):
    """Grow morphologies in vacuum.

    With diametrizer='external', we will use diameter-synthesis,
    otherwise 'M1-M5' from TNS are allowed.
    """
    global_gid = 0
    rows = []
    for mtype in tqdm(mtypes):
        tmd_parameters[mtype]["diameter_params"]["method"] = diametrizer
        tmd_distributions["mtypes"][mtype]["diameter"]["method"] = diametrizer

        if diametrizer == "external":
            external_diametrizer = build_diameters.build
        else:
            external_diametrizer = None

        gids = range(global_gid, global_gid + n_cells)
        global_gid += n_cells
        for row in Parallel(
            nb_jobs,
            verbose=joblib_verbose,
            backend="multiprocessing",
            batch_size=1 + int(len(gids) / (cpu_count() if nb_jobs == -1 else nb_jobs)),
        )(
            delayed(_grow_morphology)(
                gid,
                mtype,
                tmd_parameters[mtype],
                tmd_distributions["mtypes"][mtype],
                morphology_base_path,
                vacuum_morphology_path=vacuum_morphology_path,
                external_diametrizer=external_diametrizer,
            )
            for gid in gids
        ):
            rows.append(row)
    vacuum_synth_morphs_df = pd.concat(rows)
    return vacuum_synth_morphs_df


def plot_vacuum_morphologies(vacuum_synth_morphs_df, pdf_filename, morphology_path):
    """Plot synthesized morphologies on top of each others."""
    with PdfPages(pdf_filename) as pdf:
        for mtype in tqdm(sorted(vacuum_synth_morphs_df.mtype.unique())):
            plt.figure()
            ax = plt.gca()
            for gid in vacuum_synth_morphs_df[vacuum_synth_morphs_df.mtype == mtype].index:
                morphology = load_morphology(vacuum_synth_morphs_df.loc[gid, morphology_path])
                matplotlib_impl.plot_morph(morphology, ax, plane="zy")
            ax.set_title(mtype)
            ax.set_rasterized(True)
            plt.axis([-800, 800, -800, 2000])
            with DisableLogger():
                pdf.savefig()
            plt.close()
