"""Functions for validation of synthesis to be used by luigi tasks."""
import logging
import os
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from pyquaternion import Quaternion

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import delayed
from joblib import Parallel
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import fmin
from tqdm import tqdm
from voxcell import CellCollection

from atlas_analysis.constants import CANONICAL
from morph_validator.feature_configs import get_feature_configs
from morph_validator.plotting import get_features_df
from morph_validator.plotting import plot_violin_features
from morph_validator.spatial import relative_depth_volume
from morph_validator.spatial import sample_morph_voxel_values
from neurom import viewer

from .circuit import get_cells_between_planes
from .tools import ensure_dir


L = logging.getLogger(__name__)
matplotlib.use("Agg")


def convert_mvd3_to_morphs_df(mvd3_path, synth_output_path, ext="asc"):
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
        lambda morph: (Path(synth_output_path) / morph).with_suffix("." + ext)
    )
    cells_df["name"] = cells_df["morphology"]
    return cells_df.drop("morphology", axis=1)


def _get_features_df_all_mtypes(morphs_df, features_config, morphology_path):
    """Wrapper for morph-validator functions."""
    morphs_df_dict = {
        mtype: df[morphology_path] for mtype, df in morphs_df.groupby("mtype")
    }
    with warnings.catch_warnings():
        # Ignore some Numpy warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return get_features_df(
            morphs_df_dict, features_config, n_workers=os.cpu_count()
        )


def plot_morphometrics(
    base_morphs_df,
    comp_morphs_df,
    output_path,
    base_key="morphology_path",
    comp_key="synth_morphology_path",
    base_label="base",
    comp_label="comp",
    normalize=False,
    config_features=None,
):
    """Plot morphometrics.

    Args:
        base_morphs_df (DataFrame): base morphologies
        comp_morphs_df (DataFrame): compared morphologies
        output_path (str): path to save figures
        base_key (str): column name in the DF
        comp_key (str): column name in the DF
        base_label (str): label for the base morphologies
        comp_label (str): label for the compared morphologies
        normalize (bool): normalize data if set to True
        config_features (dict): mapping of features to plot
    """
    if config_features is None:
        config_features = get_feature_configs(config_types="synthesis")
        config_features["neurite"].update({"y_distances": ["min", "max"]})

    base_features_df = _get_features_df_all_mtypes(
        base_morphs_df, config_features, base_key
    )
    base_features_df["label"] = base_label
    comp_features_df = _get_features_df_all_mtypes(
        comp_morphs_df, config_features, comp_key
    )
    comp_features_df["label"] = comp_label

    base_features_df = base_features_df[
        base_features_df.mtype.isin(comp_features_df.mtype.unique())
    ]
    comp_features_df = comp_features_df[
        comp_features_df.mtype.isin(base_features_df.mtype.unique())
    ]

    all_features_df = pd.concat([base_features_df, comp_features_df])
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


def get_plane_rotation_matrix(plane, target=None):
    """Get basis vectors best aligned to target direction.

    We define a direct orthonormal basis of the plane (e_1, e_2) such
    that || e_2 - target || is minimal. The positive axes along the
    vectors e_1  and e_2 correspond respectively to the horizontal and
    vertical dimensions of the image.

    Args:
        plane (atlas_analysis.plane.maths.Plane): plane object
        target (list): target vector to align each plane

    Return:
        np.ndarray: rotation matrix to map VoxelData coordinates to plane coordinates
    """
    if target is None:
        target = [0, 1, 0]
    target /= np.linalg.norm(target)

    quaternion = plane.get_quaternion(CANONICAL[2])
    rotation_matrix = quaternion.rotation_matrix

    def _get_rot_matrix(angle):
        """Get rotation matrix for a given angle along [0, 0, 1]."""
        return Quaternion(axis=[0, 0, 1], angle=angle).rotation_matrix

    def _cost(angle):
        return np.linalg.norm(
            rotation_matrix.dot(
                _get_rot_matrix(angle).dot(np.array([0, 1, 0])) - target
            )
        )

    angle = fmin(_cost, 1.0, disp=False)
    return _get_rot_matrix(angle).dot(rotation_matrix)


def get_layer_info(
    layer_annotation,
    plane_origin,
    rotation_matrix,
    n_pixels=1024,
):
    """Get information to plot layers on a plane.

    Args:
        layer_annotation (VoxelData): atlas annotations with layers
        plane_origin (np.ndarray): origin of plane (Plane.point)
        rotation_matrix (3*3 np.ndarray): rotation matrix to transform from real coordinates
            to plane coordinates
        n_pixels (int): number of pixel on each axis of the plane for plotting layers
    """
    bbox = layer_annotation.bbox
    bbox_min = rotation_matrix.dot(bbox[0] - plane_origin)
    bbox_max = rotation_matrix.dot(bbox[1] - plane_origin)
    xs_plane = np.linspace(bbox_min[0], bbox_max[0], n_pixels)
    ys_plane = np.linspace(bbox_min[1], bbox_max[1], n_pixels)

    layers = np.empty([n_pixels, n_pixels])
    X = np.empty([n_pixels, n_pixels])
    Y = np.empty([n_pixels, n_pixels])

    for i, x_plane in enumerate(xs_plane):
        for j, y_plane in enumerate(ys_plane):
            X[i, j] = x_plane
            Y[i, j] = y_plane

            # transform plane coordinates into real coordinates (coordinates of VoxelData)
            point = (
                rotation_matrix.T.dot([x_plane, 0, 0])
                + rotation_matrix.T.dot([0, y_plane, 0])
                + plane_origin
            )

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
    rotation_matrix=None,
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
        # transform morphology in the plane coordinates
        morphology = morphology.transform(
            lambda p: np.dot(p - plane_left.point, rotation_matrix.T)
        )
        viewer.plot_neuron(
            ax,
            morphology,
            plane="xy",
            realistic_diameters=True,
        )


def _plot_collage(
    planes, layer_annotation=None, circuit=None, mtype=None, sample=None, n_pixels=1024
):
    """Internal plot collage for multiprocessing."""
    left_plane, right_plane = planes
    rotation_matrix = get_plane_rotation_matrix(left_plane)
    X, Y, layers = get_layer_info(
        layer_annotation, left_plane.point, rotation_matrix, n_pixels
    )

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
        rotation_matrix=rotation_matrix,
        mtype=mtype,
        sample=sample,
    )
    plt.colorbar()
    ax = plt.gca()
    ax.set_rasterized(True)
    ax.set_title("")
    return fig


def plot_collage(
    circuit,
    planes,
    layer_annotation,
    mtype,
    pdf_filename="collage.pdf",
    sample=10,
    nb_jobs=-1,
    joblib_verbose=10,
    dpi=1000,
):
    """Plot collage of an mtype and a list of planes.

    Args:
        circuit (circuit): should contain location of soma and mtypes
        planes (list): list of plane objects from atlas_analysis
        layer_annotation (VoxelData): layer annotation on atlas
        mtype (str): mtype of cells to plot
        pdf_filename (str): pdf filename
        sample (int): maximum number of cells to plot
        nb_jobs (int) : number of joblib workers
        joblib_verbose (int) verbose level of joblib
        dpi (int): dpi for pdf rendering (rasterized)
    """
    ensure_dir(pdf_filename)
    with PdfPages(pdf_filename) as pdf:
        f = partial(
            _plot_collage,
            layer_annotation=layer_annotation,
            circuit=circuit,
            mtype=mtype,
            sample=sample,
        )
        for fig in Parallel(nb_jobs, verbose=joblib_verbose)(
            delayed(f)(planes) for planes in zip(planes[:-1:3], planes[2::3])
        ):
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
