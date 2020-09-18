"""Functions for validation of synthesis to be used by luigi tasks."""
import logging
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import delayed
from joblib import Parallel
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import expm
from scipy.optimize import fmin
from tqdm import tqdm
from voxcell import CellCollection

from atlas_analysis.constants import CANONICAL
from atlas_analysis.planes.maths import Plane
from morph_validator.feature_configs import get_feature_configs
from morph_validator.plotting import get_features_df
from morph_validator.plotting import plot_violin_features
from morph_validator.spatial import relative_depth_volume
from morph_validator.spatial import sample_morph_voxel_values
from neurom import viewer

from .circuit_slicing import get_cells_between_planes
from .tools import ensure_dir


L = logging.getLogger(__name__)
matplotlib.use("Agg")


def convert_mvd3_to_morphs_df(mvd3_path, synth_output_path, ext=".asc"):
    """Convert the list of morphologies from mvd3 to morphology dataframe.

    Args:
        mvd3_path (str): path to mvd3 (somata) file
        synth_output_path (str): path to morphology files
        ext( str): extension of morphology files

    Returns:
        DataFrame: morphology dataframe
    """
    cells_df = CellCollection.load_mvd3(mvd3_path).as_dataframe()
    cells_df["synth_morphology_path"] = cells_df["morphology"].apply(
        lambda morph: (Path(synth_output_path) / morph).with_suffix(ext)
    )
    cells_df["name"] = cells_df["morphology"]
    return cells_df.drop("morphology", axis=1)


def _get_features_df_all_mtypes(morphs_df, features_config, morphology_path):
    """Wrapper for morph-validator functions."""
    morphs_df_dict = {
        mtype: df[morphology_path] for mtype, df in morphs_df.groupby("mtype")
    }
    return get_features_df(morphs_df_dict, features_config, n_workers=os.cpu_count())


def plot_morphometrics(
    morphs_df,
    synth_morphs_df,
    output_path,
    bio_key="morphology_path",
    synth_key="synth_morphology_path",
    normalize=False,
):
    """Plot morphometrics.

    Args:
        morphs_df (DataFrame): reconstructed morphologies
        synth_morphs_df (DataFrame): synthesized morphologies
        output_path (str): path to save figures
        bio_key (str): column name in the DF
        synth_key (str): column name in the DF
        normalize (bool): normalize data if set to True
    """
    config_features = get_feature_configs(config_types="synthesis")
    config_features["neurite"] = {"y_distances": ["min", "max"]}

    bio_features_df = _get_features_df_all_mtypes(morphs_df, config_features, bio_key)
    bio_features_df["label"] = "bio"
    synth_features_df = _get_features_df_all_mtypes(
        synth_morphs_df, config_features, synth_key
    )
    synth_features_df["label"] = "synth"

    bio_features_df = bio_features_df[
        bio_features_df.mtype.isin(synth_features_df.mtype.unique())
    ]
    synth_features_df = synth_features_df[
        synth_features_df.mtype.isin(bio_features_df.mtype.unique())
    ]

    all_features_df = pd.concat([bio_features_df, synth_features_df])
    ensure_dir(output_path)
    plot_violin_features(
        all_features_df,
        ["basal_dendrite", "apical_dendrite"],
        output_dir=Path(output_path),
        bw=0.1,
        normalize=normalize,
    )


def _get_depths_df(circuit, mtype, sample, voxeldata, sample_distance):
    """Create dataframe with depths data for violin plots."""
    out_of_bounds_value = np.nan
    gids = circuit.cells.ids(group={"mtype": mtype}, sample=sample)

    point_depths = defaultdict(list)
    for gid in gids:
        morphology = circuit.morph.get(gid, transform=True, source="ascii")
        point_depth_tmp = sample_morph_voxel_values(
            morphology, sample_distance, voxeldata, out_of_bounds_value
        )
        for neurite_type, data in point_depth_tmp.items():
            point_depths[neurite_type.name] += data.tolist()

    df = pd.DataFrame.from_dict(point_depths, orient="index").T
    return df.melt(var_name="neurite_type", value_name="y").dropna()


def _plot_layers(x_pos, atlas, ax):
    """Plot the layers at position x."""
    layers = np.arange(1, 7)
    bbox = atlas.load_data("[PH]1").bbox
    z_scan = np.linspace(bbox[0, 2] + 1, bbox[1, 2] - 1, 20)
    # z_scan = np.linspace(1100, 1300, 100)
    x_pos = 0  # 0.5 * (bbox[1, 0] + bbox[0, 0])
    y_pos = 0  # 0.5 * (bbox[1, 1] + bbox[0, 1])

    all_layer_bounds = []
    for layer in layers:
        layer_bounds = []
        ph_layer = atlas.load_data(f"[PH]{layer}")
        for z in z_scan:
            layer_bounds.append(list(ph_layer.lookup([x_pos, y_pos, z])))
        all_layer_bounds.append(np.array(layer_bounds))

    for layer, layer_bounds in enumerate(all_layer_bounds):
        ax.fill_between(
            z_scan,
            layer_bounds[:, 0],
            layer_bounds[:, 1],
            alpha=0.5,
            label=f"layer {layer+1}",
        )


def _plot_density_profile(
    mtype, circuit=None, x_pos=None, sample=None, voxeldata=None, sample_distance=None
):
    """Plot density profile of an mtype."""
    fig = plt.figure()
    ax = plt.gca()
    _plot_layers(x_pos, circuit.atlas, ax)
    try:
        plot_df = _get_depths_df(circuit, mtype, sample, voxeldata, sample_distance)
        sns.violinplot(x="neurite_type", y="y", data=plot_df, ax=ax, bw=0.1)
        ax.legend(loc="best")
    except Exception:  # pylint: disable=broad-except
        ax.text(
            0.5,
            0.5,
            "ERROR WHEN GETTING POINT DEPTHS",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    fig.suptitle(mtype)
    return fig


def plot_density_profiles(
    circuit, sample, region, sample_distance, output_path, nb_jobs=-1
):
    """Plot density profiles for all mtypes.

    WIP function, waiting on complete atlas to update.
    """
    voxeldata = relative_depth_volume(circuit.atlas, in_region=region, relative=False)
    x_pos = 0

    ensure_dir(output_path)
    with PdfPages(output_path) as pdf:
        f = partial(
            _plot_density_profile,
            circuit=circuit,
            x_pos=x_pos,
            sample=sample,
            voxeldata=voxeldata,
            sample_distance=sample_distance,
        )
        for fig in Parallel(nb_jobs)(
            delayed(f)(mtype) for mtype in tqdm(sorted(circuit.cells.mtypes))
        ):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _plot_cells(circuit, mtype, sample, ax):
    """Plot cells for collage."""
    max_sample = (circuit.cells.get(properties="mtype") == mtype).sum()
    if sample > max_sample:
        L.warning(
            "The sample value is set to '%s' for the type %s because there are no more "
            "cells available of that type",
            max_sample,
            mtype,
        )
        sample = max_sample
    gids = circuit.cells.ids(group={"mtype": mtype}, sample=sample)

    for gid in gids:
        morphology = circuit.morph.get(gid, transform=True, source="ascii")
        viewer.plot_neuron(
            ax,
            morphology,
            plane="zy",
            realistic_diameters=True,
        )


def _plot_collage_O1(
    mtype, circuit=None, figsize=None, x_pos=None, ax_limit=None, sample=None
):
    """Plot collage for a given mtype (for multiprocessing)."""
    fig = plt.figure(mtype, figsize=figsize)
    ax = plt.gca()
    try:
        _plot_layers(x_pos, circuit.atlas, ax)
    except Exception:  # pylint: disable=broad-except
        L.error("Unable to plot the layers for the type '%s'", mtype)
    try:
        _plot_cells(circuit, mtype, sample, ax)
    except Exception:  # pylint: disable=broad-except
        L.error("Unable to plot the cells for the type '%s'", mtype)
    plt.axis(ax_limit)

    ax.set_rasterized(True)
    ax.legend(loc="best")
    fig.suptitle(mtype)
    return fig


def plot_collage_O1(circuit, sample, output_path, mtypes=None, nb_jobs=-1):
    """Plot collage for all mtypes.

    Args:
        circuit (circuit): circuit
        sample (int): number of cells to plot (None means all available)
        output_path (str): path to save pdf file
    """
    x_pos = 0
    atlas_bbox = circuit.atlas.load_data("[PH]1").bbox
    x_min = atlas_bbox[0, 2]
    x_max = atlas_bbox[1, 2]
    y_min = atlas_bbox[0, 1]
    y_max = atlas_bbox[1, 1]

    dx = x_max - x_min
    x_min -= 0.1 * dx
    x_max += 0.1 * dx

    dy = y_max - y_min
    y_min -= 0.1 * dy
    y_max += 0.1 * dy

    figsize = (0.01 * (x_max - x_min), 0.01 * (y_max - y_min))
    ax_limit = [x_min, x_max, y_min, y_max]

    if mtypes is None:
        mtypes = sorted(list(circuit.cells.mtypes))

    ensure_dir(output_path)
    with PdfPages(output_path) as pdf:
        f = partial(
            _plot_collage_O1,
            circuit=circuit,
            figsize=figsize,
            x_pos=x_pos,
            ax_limit=ax_limit,
            sample=sample,
        )
        for fig in Parallel(nb_jobs)(
            delayed(f)(mtype) for mtype in tqdm(sorted(circuit.cells.mtypes))
        ):
            pdf.savefig(fig, bbox_inches="tight", dpi=100)
            plt.close(fig)


def get_aligned_basis(plane, target=None):
    """Get basis vectors best aligned to target direction."""
    if target is None:
        target = [0, 0, 1]
    plane_cls = Plane.from_quaternion(plane[:3], plane[3:])
    plane_basis = plane_cls.get_basis()

    quaternion = plane_cls.get_quaternion(CANONICAL[2])
    r_axis = quaternion.rotate(CANONICAL[2])

    def get_rot_matrix(angle):
        """get rotation matrix for a given angle."""
        r_basis = np.array(
            [
                [0, -r_axis[2], r_axis[1]],
                [r_axis[2], 0, -r_axis[0]],
                [-r_axis[1], r_axis[0], 0],
            ]
        )
        return expm(angle * r_basis)

    def cost(angle):
        return np.linalg.norm(target - get_rot_matrix(angle).dot(plane_basis[0]))

    angle = fmin(cost, 1, disp=False)
    rot_matrix = get_rot_matrix(angle)

    plane_basis[0] = rot_matrix.dot(plane_basis[0])
    plane_basis[1] = rot_matrix.dot(plane_basis[1])
    rotation_matrix = rot_matrix.dot(plane_cls.get_quaternion().rotation_matrix)
    return plane_basis, rotation_matrix


def get_layer_info(
    layer_annotation,
    plane,
    plane_basis,
    n_pixels=512,
    limits=None,
):
    """Get information to plot layers on a plane."""
    if limits is None:
        limits = [-5000, 8000, -9000, 2000]
    xs_plane = np.linspace(limits[0], limits[1], n_pixels)
    ys_plane = np.linspace(limits[2], limits[3], n_pixels)

    layers = np.empty([n_pixels, n_pixels])
    X = np.empty([n_pixels, n_pixels])
    Y = np.empty([n_pixels, n_pixels])

    for i, x_plane in enumerate(xs_plane):
        for j, y_plane in enumerate(ys_plane):
            point = x_plane * plane_basis[0] + y_plane * plane_basis[1] + plane[:3]
            # this plots the z-x coordinates, we would eventualy like to plot in plane coordinates
            X[i, j] = point[2]  # x_plane
            Y[i, j] = point[0]  # y_plane
            layers[i, j] = int(
                layer_annotation.lookup(np.array([point]), outer_value=-1)
            )
    layers[layers < 1] = np.nan
    return X, Y, layers


def plot_cells(
    ax,
    circuit,
    plane_left,
    plane_right,
    mtype=None,
    sample=10,
):
    """Plot cells for collage."""
    cells = circuit.cells.get({"mtype": mtype})
    if mtype is not None:
        cells = cells[cells.mtype == mtype]

    if len(cells) == 0:
        raise Exception("no cells of that mtype")

    gids = get_cells_between_planes(cells, plane_left, plane_right).index
    for gid in gids[:sample]:
        morphology = circuit.morph.get(gid, transform=True, source="ascii")
        viewer.plot_neuron(
            ax,
            morphology,
            plane="zx",
            realistic_diameters=True,
        )


def _plot_collage(
    plane_id, planes=None, layer_annotation=None, circuit=None, mtype=None, sample=None
):
    """Internal plot collage for multiprocessing."""
    left_plane = planes[2 * plane_id]
    right_plane = planes[2 * plane_id + 1]

    plane_basis, _ = get_aligned_basis(left_plane)
    X, Y, layers = get_layer_info(layer_annotation, left_plane, plane_basis)

    fig = plt.figure()
    plt.contourf(
        X,
        Y,
        layers,
        cmap="tab10",
        vmin=0,
        vmax=10,
        levels=[1, 2, 3, 4, 5, 6],
        alpha=0.5,
    )
    plot_cells(
        plt.gca(),
        circuit,
        left_plane,
        right_plane,
        mtype=mtype,
        sample=sample,
    )
    plt.colorbar()
    return fig


def plot_collage(
    circuit,
    planes,
    layer_annotation,
    mtype,
    output_path="collage.pdf",
    sample=10,
    nb_jobs=-1,
):
    """Plot collage of an mtyp and a list of planes."""
    plane_ids = np.arange(int(len(planes) / 2) - 1)

    ensure_dir(output_path)
    with PdfPages(output_path) as pdf:
        f = partial(
            _plot_collage,
            planes=planes,
            layer_annotation=layer_annotation,
            circuit=circuit,
            mtype=mtype,
            sample=sample,
        )
        for fig in Parallel(nb_jobs)(
            delayed(f)(plane_id) for plane_id in tqdm(plane_ids)
        ):
            pdf.savefig(fig, bbox_inches="tight", dpi=100)
            plt.close(fig)
