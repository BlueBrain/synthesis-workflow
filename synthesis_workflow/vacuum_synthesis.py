"""Functions for synthesis to be used by luigi tasks."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from morphio.mut import Morphology
from neurom import load_neuron
from neurom import viewer
from tns import NeuronGrower

from . import STR_TO_TYPES
from .synthesis import get_max_len


def _grow_morphology(
    gid, mtype, tmd_parameters, tmd_distributions, morphology_base_path
):
    """Grow single morphology for parallel computations."""

    name = f"vacuum_{gid}.asc"
    morphology_path = morphology_base_path / name
    vacuum_synth_morphs_df = pd.DataFrame()
    np.random.seed(gid)

    grower = NeuronGrower(
        input_parameters=tmd_parameters,
        input_distributions=tmd_distributions,
        external_diametrizer=None,
    )
    grower.grow()
    grower.neuron.write(morphology_path)

    vacuum_synth_morphs_df.loc[gid, "name"] = name
    vacuum_synth_morphs_df.loc[gid, "mtype"] = mtype
    vacuum_synth_morphs_df.loc[gid, "vacuum_morphology_path"] = morphology_path
    # vacuum_synth_morphs_df.loc[gid, 'apical_point'] = grower.apical_points

    return vacuum_synth_morphs_df


def grow_vacuum_morphologies(
    mtypes,
    n_cells,
    tmd_parameters,
    tmd_distributions,
    morphology_base_path,
    joblib_verbose=0,
    nb_jobs=1,
):
    """Grow morphologies in vacuum."""

    global_gid = 0
    vacuum_synth_morphs_df = pd.DataFrame()
    for mtype in tqdm(mtypes):
        # no need to realistic diameters here, using internal TNS diametrizer
        tmd_parameters[mtype]["diameter_params"]["method"] = "M1"
        tmd_distributions["mtypes"][mtype]["diameter"]["method"] = "M1"

        gids = range(global_gid, global_gid + n_cells)
        global_gid += n_cells
        for row in Parallel(nb_jobs, verbose=joblib_verbose)(
            delayed(_grow_morphology)(
                gid,
                mtype,
                tmd_parameters[mtype],
                tmd_distributions["mtypes"][mtype],
                morphology_base_path,
            )
            for gid in gids
        ):
            vacuum_synth_morphs_df = vacuum_synth_morphs_df.append(row)
    return vacuum_synth_morphs_df


def plot_vacuum_morphologies(
    vacuum_synth_morphs_df, pdf_filename, morphology_path, mean_lengths
):
    """Plot synthesized morphologies on top of each others."""
    colors = {"apical": "m", "basal": "r", "axon": "b"}

    with PdfPages(pdf_filename) as pdf:
        for mtype in tqdm(sorted(vacuum_synth_morphs_df.mtype.unique())):
            plt.figure()
            ax = plt.gca()
            for gid in vacuum_synth_morphs_df[
                vacuum_synth_morphs_df.mtype == mtype
            ].index:
                morphology = load_neuron(
                    vacuum_synth_morphs_df.loc[gid, morphology_path]
                )
                viewer.plot_neuron(ax, morphology, plane="zy")
                morph = Morphology(vacuum_synth_morphs_df.loc[gid, morphology_path])
                for neurite in morph.root_sections:
                    if neurite.type == STR_TO_TYPES["apical"]:
                        max_len = get_max_len(neurite)
                        ax.axhline(max_len, c="0.5", lw=0.5)
            for neurite_type in mean_lengths:
                if mtype in mean_lengths[neurite_type]:
                    ax.axhline(
                        mean_lengths[neurite_type][mtype], c=colors[neurite_type]
                    )
            ax.set_title(mtype)
            ax.set_rasterized(True)
            plt.axis([-800, 800, -800, 2000])
            pdf.savefig()
            plt.close()
