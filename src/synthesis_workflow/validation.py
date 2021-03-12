"""Functions for validation of synthesis to be used by luigi tasks."""
import json
import glob
import logging
import os
import re
import warnings
from collections import defaultdict
from collections import namedtuple
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import delayed
from joblib import Parallel
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from pyquaternion import Quaternion
from scipy.optimize import fmin

from atlas_analysis.constants import CANONICAL
from bluepy.v2 import Circuit
from morph_validator.feature_configs import get_feature_configs
from morph_validator.plotting import get_features_df
from morph_validator.plotting import plot_violin_features
from morph_validator.spatial import relative_depth_volume
from morph_validator.spatial import sample_morph_voxel_values
from morphio.mut import Morphology
from neurom import viewer
from neurom.apps import morph_stats
from neurom.core.dataformat import COLS
from neurom.io import load_neurons
from region_grower.atlas_helper import AtlasHelper
from region_grower.modify import scale_target_barcode
from tmd.io.io import load_population
from tns import NeuronGrower
from voxcell import CellCollection
from voxcell.exceptions import VoxcellError

from synthesis_workflow.circuit import get_cells_between_planes
from synthesis_workflow.fit_utils import clean_outliers
from synthesis_workflow.fit_utils import get_path_distances
from synthesis_workflow.fit_utils import get_path_distance_from_extent
from synthesis_workflow.fit_utils import get_projections
from synthesis_workflow.tools import ensure_dir
from synthesis_workflow.utils import DisableLogger


L = logging.getLogger(__name__)
matplotlib.use("Agg")


VacuumCircuit = namedtuple("VacuumCircuit", ["cells", "morphs_df", "morphology_path"])

SYNTH_MORPHOLOGY_PATH = "synth_morphology_path"


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
    cells_df[SYNTH_MORPHOLOGY_PATH] = cells_df["morphology"].apply(
        lambda morph: (Path(synth_output_path) / morph).with_suffix("." + ext)
    )
    cells_df["name"] = cells_df["morphology"]
    return cells_df.drop("morphology", axis=1)


def _get_features_df_all_mtypes(morphs_df, features_config, morphology_path):
    """Wrapper for morph-validator functions."""
    morphs_df_dict = {mtype: df[morphology_path] for mtype, df in morphs_df.groupby("mtype")}
    with warnings.catch_warnings():
        # Ignore some Numpy warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return get_features_df(morphs_df_dict, features_config, n_workers=os.cpu_count())


def plot_morphometrics(
    base_morphs_df,
    comp_morphs_df,
    output_path,
    base_key="morphology_path",
    comp_key=SYNTH_MORPHOLOGY_PATH,
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
        # config_features["neurite"].update({"y_distances": ["min", "max"]})

    base_features_df = _get_features_df_all_mtypes(base_morphs_df, config_features, base_key)
    base_features_df["label"] = base_label
    comp_features_df = _get_features_df_all_mtypes(comp_morphs_df, config_features, comp_key)
    comp_features_df["label"] = comp_label

    base_features_df = base_features_df[
        base_features_df.mtype.isin(comp_features_df.mtype.unique())
    ]
    comp_features_df = comp_features_df[
        comp_features_df.mtype.isin(base_features_df.mtype.unique())
    ]

    all_features_df = pd.concat([base_features_df, comp_features_df])
    ensure_dir(output_path)
    with DisableLogger():
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
    return df.melt(var_name="neurite_type", value_name="y").dropna().sort_values("neurite_type")


def _get_vacuum_depths_df(circuit, mtype):
    """Create dataframe with depths data for violin plots."""
    morphs_df = circuit.morphs_df
    path = circuit.morphology_path
    cells = morphs_df.loc[morphs_df["mtype"] == mtype, path]
    point_depths = defaultdict(list)
    for cell_path in cells:
        morph = Morphology(cell_path)
        for i in morph.iter():
            point_depths[i.type.name] += i.points[COLS.Y].tolist()

    df = pd.DataFrame.from_dict(point_depths, orient="index").T
    return df.melt(var_name="neurite_type", value_name="y").dropna().sort_values("neurite_type")


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
    try:
        if isinstance(circuit, Circuit):
            _plot_layers(x_pos, circuit.atlas, ax)
            plot_df = _get_depths_df(circuit, mtype, sample, voxeldata, sample_distance)
            ax.legend(loc="best")
        elif isinstance(circuit, VacuumCircuit):
            plot_df = _get_vacuum_depths_df(circuit, mtype)

        with DisableLogger():
            sns.violinplot(x="neurite_type", y="y", data=plot_df, ax=ax, bw=0.1)
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


def plot_density_profiles(circuit, sample, region, sample_distance, output_path, nb_jobs=-1):
    """Plot density profiles for all mtypes.

    WIP function, waiting on complete atlas to update.
    """
    if not region or region == "in_vacuum":
        voxeldata = None
    else:
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
        for fig in Parallel(nb_jobs)(delayed(f)(mtype) for mtype in sorted(circuit.cells.mtypes)):
            with DisableLogger():
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


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
            rotation_matrix.dot(_get_rot_matrix(angle).dot(np.array([0, 1, 0])) - target)
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

    X = np.repeat(xs_plane.reshape((1, -1)), n_pixels, axis=0).T
    Y = np.repeat(ys_plane.reshape((1, -1)), n_pixels, axis=0)
    rot_T = rotation_matrix.T

    xs_plane_size = xs_plane.size
    ys_plane_size = ys_plane.size
    x_vec = xs_plane.copy()
    x_vec.resize((3, xs_plane_size), refcheck=False)
    x_vec = x_vec.T

    y_vec = ys_plane.copy()
    y_vec.resize((3, ys_plane_size), refcheck=False)
    y_vec = y_vec.T[:, [1, 0, 2]]  # pylint: disable=unsubscriptable-object

    x_rot = np.einsum("ij, kj", rot_T, x_vec).T
    y_rot = np.einsum("ij, kj", rot_T, y_vec).T

    y_final = np.repeat(y_rot[np.newaxis, ...], xs_plane_size, axis=0)
    x_final = np.repeat(x_rot, ys_plane_size, axis=0).reshape(y_final.shape)

    points = x_final + y_final + plane_origin

    layers = layer_annotation.lookup(points, outer_value=-1).astype(float)
    layers[layers < 1] = np.nan
    return X, Y, layers


def get_y_info(atlas, plane_origin, rotation_matrix, n_pixels=64):
    """Get direction of y axis on a grid on the atlas planes."""
    bbox = atlas.orientations.bbox
    bbox_min = rotation_matrix.dot(bbox[0] - plane_origin)
    bbox_max = rotation_matrix.dot(bbox[1] - plane_origin)
    xs_plane = np.linspace(bbox_min[0], bbox_max[0], n_pixels)
    ys_plane = np.linspace(bbox_min[1], bbox_max[1], n_pixels)

    orientation_u = np.zeros([n_pixels, n_pixels])
    orientation_v = np.zeros([n_pixels, n_pixels])

    X = np.repeat(xs_plane.reshape((1, -1)), n_pixels, axis=0).T
    Y = np.repeat(ys_plane.reshape((1, -1)), n_pixels, axis=0)
    rot_T = rotation_matrix.T
    for i, x_plane in enumerate(xs_plane):
        for j, y_plane in enumerate(ys_plane):
            # transform plane coordinates into real coordinates (coordinates of VoxelData)
            point = rot_T.dot([x_plane, 0, 0]) + rot_T.dot([0, y_plane, 0]) + plane_origin
            try:
                orientation = atlas.lookup_orientation(point)
                if orientation[0] != 0.0 and orientation[1] != 1.0:
                    orientation_u[i, j], orientation_v[i, j] = rotation_matrix.dot(orientation)[:2]
            except VoxcellError:
                pass
    return X, Y, orientation_u, orientation_v


def plot_cells(
    ax,
    circuit,
    plane_left,
    plane_right,
    rotation_matrix=None,
    mtype=None,
    sample=10,
    atlas=None,
    plot_neuron_kwargs=None,
):
    """Plot cells for collage."""
    if mtype is not None:
        cells = circuit.cells.get({"mtype": mtype})
    else:
        cells = circuit.cells.get()

    if len(cells) == 0:
        raise Exception(f"No cell of that mtype ({mtype})")

    if plot_neuron_kwargs is None:
        plot_neuron_kwargs = {}

    gids = get_cells_between_planes(cells, plane_left, plane_right).index[:sample]

    if atlas is not None:
        vec = [0, 1, 0]
        all_pos_orig = cells.loc[gids, ["x", "y", "z"]].values
        all_orientations = atlas.orientations.lookup(all_pos_orig)
        all_lookups = np.einsum("ijk, k", all_orientations, vec)
        all_pos_final = all_pos_orig + all_lookups * 300
        all_dist_plane_orig = all_pos_orig - plane_left.point
        all_dist_plane_final = all_pos_final - plane_left.point
        all_pos_orig_plane_coord = np.tensordot(all_dist_plane_orig, rotation_matrix.T, axes=1)
        all_pos_final_plane_coord = np.tensordot(all_dist_plane_final, rotation_matrix.T, axes=1)

    for num, gid in enumerate(gids):
        morphology = circuit.morph.get(gid, transform=True, source="ascii")

        def _to_plane_coord(p):
            return np.dot(p - plane_left.point, rotation_matrix.T)

        # transform morphology in the plane coordinates
        morphology = morphology.transform(_to_plane_coord)

        if atlas is not None:
            plt.plot(
                [all_pos_orig_plane_coord[num, 0], all_pos_final_plane_coord[num, 0]],
                [all_pos_orig_plane_coord[num, 1], all_pos_final_plane_coord[num, 1]],
                c="0.5",
                lw=0.2,
            )

        viewer.plot_neuron(ax, morphology, plane="xy", **plot_neuron_kwargs)


# pylint: disable=too-many-arguments
def _plot_collage(
    planes,
    layer_annotation=None,
    circuit=None,
    mtype=None,
    sample=None,
    n_pixels=1024,
    atlas=None,
    n_pixels_y=64,
    with_y_field=True,
    with_cells=True,
    plot_neuron_kwargs=None,
):
    """Internal plot collage for multiprocessing."""
    if with_y_field and atlas is None:
        raise Exception("Please provide an atlas with option with_y_field=True")

    left_plane, right_plane = planes
    rotation_matrix = get_plane_rotation_matrix(left_plane)
    X, Y, layers = get_layer_info(layer_annotation, left_plane.point, rotation_matrix, n_pixels)

    fig = plt.figure()
    plt.imshow(
        layers.T,
        extent=[X[0, 0], X[-1, 0], Y[0, 0], Y[0, -1]],
        aspect="auto",
        origin="lower",
        cmap=matplotlib.colors.ListedColormap(["C0", "C1", "C2", "C3", "C4", "C5"]),
        alpha=0.3,
    )

    plt.colorbar()
    if with_cells:
        plot_cells(
            plt.gca(),
            circuit,
            left_plane,
            right_plane,
            rotation_matrix=rotation_matrix,
            mtype=mtype,
            sample=sample,
            atlas=atlas,
            plot_neuron_kwargs=plot_neuron_kwargs,
        )

    if with_y_field:
        # note: some of these parameters are harcoded for NCx plot, adjust as needed
        X_y, Y_y, orientation_u, orientation_v = get_y_info(
            atlas, left_plane.point, rotation_matrix, n_pixels_y
        )
        plt.quiver(
            X_y,
            Y_y,
            orientation_u * 100,
            orientation_v * 100,
            width=0.0002,
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_rasterized(True)
    ax.set_title("")

    return fig


# pylint: disable=too-many-arguments
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
    n_pixels=1024,
    with_y_field=True,
    n_pixels_y=64,
    plot_neuron_kwargs=None,
    with_cells=True,
):
    """Plot collage of an mtype and a list of planes.

    Args:
        circuit (circuit): should contain location of soma and mtypes
        planes (list): list of plane objects from atlas_analysis
        layer_annotation (VoxelData): layer annotation on atlas
        mtype (str): mtype of cells to plot
        pdf_filename (str): pdf filename
        sample (int): maximum number of cells to plot
        nb_jobs (int): number of joblib workers
        joblib_verbose (int): verbose level of joblib
        dpi (int): dpi for pdf rendering (rasterized)
        n_pixels (int): number of pixels for plotting layers
        with_y_field (bool): plot y field
        n_pixels_y (int): number of pixels for plotting y field
        plot_neuron_kwargs (dict): dict given to ``neurom.viewer.plot_neuron`` as kwargs
        with_cells (bool): plot cells or not
    """
    atlas = AtlasHelper(circuit.atlas)

    ensure_dir(pdf_filename)
    with PdfPages(pdf_filename) as pdf:
        f = partial(
            _plot_collage,
            layer_annotation=layer_annotation,
            circuit=circuit,
            mtype=mtype,
            sample=sample,
            atlas=atlas,
            n_pixels=n_pixels,
            n_pixels_y=n_pixels_y,
            with_y_field=with_y_field,
            plot_neuron_kwargs=plot_neuron_kwargs,
            with_cells=with_cells,
        )
        for fig in Parallel(nb_jobs, verbose=joblib_verbose)(
            delayed(f)(planes) for planes in zip(planes[:-1:3], planes[2::3])
        ):
            with DisableLogger():
                pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)


def _generate_synthetic_random_population(
    dir_path, nb, proj_min, proj_max, tmd_parameters, tmd_distributions
):
    """Generate a synthetic population with random projections."""
    files = []
    y_synth = []
    slope = tmd_parameters["context_constraints"]["apical"]["extent_to_target"]["slope"]
    intercept = tmd_parameters["context_constraints"]["apical"]["extent_to_target"]["intercept"]
    for i in range(nb):
        tmp_name = str((Path(dir_path) / str(i)).with_suffix(".h5"))
        files.append(tmp_name)
        projection = np.random.randint(proj_min, proj_max)
        y_synth.append(projection)
        target_path_distance = get_path_distance_from_extent(slope, intercept, projection)
        tmd_parameters["apical"].update(
            {
                "modify": {
                    "funct": scale_target_barcode,
                    "kwargs": {"target_path_distance": target_path_distance},
                }
            }
        )
        grower = NeuronGrower(
            input_parameters=tmd_parameters,
            input_distributions=tmd_distributions,
        )
        grower.grow()
        grower.neuron.write(tmp_name)

    return files, y_synth


def _get_fit_population(mtype, files, outlier_percentage, tmd_parameters, tmd_distributions):
    """Get projections and path lengths of a given and a synthetic population."""
    # Load biological neurons
    return_error = (mtype, None, None, None, None, None, None)
    if len(files) > 0:
        input_population = load_population(files)
    else:
        return return_error + (f"No file to load for mtype='{mtype}'",)
    if (
        tmd_parameters.get("context_constraints", {}).get("apical", {}).get("extent_to_target")
        is None
    ):
        return return_error + (f"No fit for mtype='{mtype}'",)

    # Get X and Y from biological population
    x = get_path_distances(input_population)
    y = get_projections(input_population)
    x_clean, y_clean = clean_outliers(x, y, outlier_percentage)

    # Create synthetic neuron population
    tmd_distributions["diameter"]["method"] = "M1"
    tmd_parameters["diameter_params"]["method"] = "M1"

    with TemporaryDirectory() as tmpdir:
        neuron_paths, y_synth = _generate_synthetic_random_population(
            tmpdir, 20, y.min(), y.max(), tmd_parameters, tmd_distributions
        )
        synthetic_population = load_population(neuron_paths)

    # Get X and Y from synthetic population
    x_synth = get_path_distances(synthetic_population)

    return mtype, x, y, x_clean, y_clean, x_synth, y_synth, None


def plot_path_distance_fits(
    tmd_parameters_path,
    tmd_distributions_path,
    morphs_df_path,
    morphology_path,
    output_path,
    mtypes=None,
    outlier_percentage=90,
    nb_jobs=-1,
):
    """Plot path-distance fits."""
    # Read TMD parameters
    with open(tmd_parameters_path) as f:
        tmd_parameters = json.load(f)

    # Read TMD distributions
    with open(tmd_distributions_path) as f:
        tmd_distributions = json.load(f)

    # Read morphology DataFrame
    morphs_df = pd.read_csv(morphs_df_path)

    if mtypes is None:
        mtypes = sorted(
            [
                mtype
                for mtype in morphs_df.mtype.unique().tolist()
                if tmd_parameters.get(mtype, {})
                .get("context_constraints", {})
                .get("apical", {})
                .get("extent_to_target")
                is not None
            ]
        )

    # Build the file list for each mtype
    file_lists = [
        (mtype, morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list())
        for mtype in mtypes
    ]

    L.debug("Number of files: %s", [(t, len(f)) for t, f in file_lists])

    ensure_dir(output_path)
    with PdfPages(output_path) as pdf:
        for mtype, x, y, x_clean, y_clean, x_synth, y_synth, msg in Parallel(nb_jobs)(
            delayed(_get_fit_population)(
                mtype,
                files,
                outlier_percentage,
                tmd_parameters[mtype],
                tmd_distributions["mtypes"][mtype],
            )
            for mtype, files in file_lists
        ):
            if all(i is None for i in [x, y, x_clean, y_clean, x_synth, y_synth]):
                L.warning(msg)
                continue
            fig = plt.figure()

            # Plot points
            plt.scatter(x, y, c="red", label="Outliers")
            plt.scatter(x_clean, y_clean, c="blue", label="Clean points")
            plt.scatter(x_synth, y_synth, c="green", label="Synthetized points")

            try:
                # Plot fit function
                plt.plot(
                    [
                        get_path_distance_from_extent(
                            tmd_parameters[mtype]["context_constraints"]["apical"][
                                "extent_to_target"
                            ]["slope"],
                            tmd_parameters[mtype]["context_constraints"]["apical"][
                                "extent_to_target"
                            ]["intercept"],
                            i,
                        )
                        for i in y
                    ],
                    y,
                    label="Clean fit",
                )
            except AttributeError:
                L.error("Could not plot the fit for %s", mtype)

            ax = plt.gca()
            ax.legend(loc="best")
            fig.suptitle(mtype)
            plt.xlabel("Path distance")
            plt.ylabel("Projection")
            with DisableLogger():
                pdf.savefig(fig, bbox_inches="tight", dpi=100)
            plt.close(fig)


def parse_log(
    log_file,
    neuron_type_position_regex,
    default_scale_regex,
    target_scale_regex,
    neurite_hard_limit_regex,
):
    """Parse log file and return a DataFrame with data."""
    # pylint: disable=too-many-locals

    def _search(pattern, line, data):
        group = re.search(pattern, line)
        if group:
            groups = group.groups()
            new_data = json.loads(groups[1])
            new_data["worker_task_id"] = int(groups[0])
            data.append(new_data)

    # List log files
    log_files = glob.glob(log_file + "/*")

    # Read log file
    all_lines = []
    for file in log_files:
        with open(file) as f:
            lines = f.readlines()
            all_lines.extend(lines)

    # Get data from log
    neuron_type_position_data = []
    default_scale_data = []
    target_scale_data = []
    neurite_hard_limit_data = []
    for line in all_lines:
        _search(neuron_type_position_regex, line, neuron_type_position_data)
        _search(default_scale_regex, line, default_scale_data)
        _search(target_scale_regex, line, target_scale_data)
        _search(neurite_hard_limit_regex, line, neurite_hard_limit_data)
    # Format data
    neuron_type_position_df = pd.DataFrame(neuron_type_position_data)
    default_scale_df = pd.DataFrame(default_scale_data)
    target_scale_df = pd.DataFrame(target_scale_data)
    neurite_hard_limit_df = pd.DataFrame(neurite_hard_limit_data)

    def _pos_to_xyz(df, drop=True):
        df[["x", "y", "z"]] = pd.DataFrame(df["position"].values.tolist(), index=df.index)
        if drop:
            df.drop(columns=["position"], inplace=True)

    # Format positions
    _pos_to_xyz(neuron_type_position_df)

    def _compute_min_max_scales(df, key, name_min, name_max):
        groups = df.groupby("worker_task_id")
        neurite_hard_min = groups[key].min().rename(name_min).reset_index()
        neurite_hard_max = groups[key].max().rename(name_max).reset_index()
        return neurite_hard_min, neurite_hard_max

    # Compute min/max hard limit scales
    neurite_hard_min, neurite_hard_max = _compute_min_max_scales(
        neurite_hard_limit_df, "scale", "hard_min_scale", "hard_max_scale"
    )
    default_min, default_max = _compute_min_max_scales(
        default_scale_df, "scaling_ratio", "default_min_scale", "default_max_scale"
    )
    target_min, target_max = _compute_min_max_scales(
        target_scale_df, "scaling_ratio", "target_min_scale", "target_max_scale"
    )

    # Merge data
    result_data = neuron_type_position_df
    result_data = result_data.merge(
        default_min, how="left", on="worker_task_id", suffixes=("", "_default_min")
    )
    result_data = result_data.merge(
        default_max, how="left", on="worker_task_id", suffixes=("", "_default_max")
    )
    result_data = result_data.merge(
        target_min, how="left", on="worker_task_id", suffixes=("", "_target_min")
    )
    result_data = result_data.merge(
        target_max, how="left", on="worker_task_id", suffixes=("", "_target_max")
    )
    result_data = result_data.merge(
        neurite_hard_min,
        how="left",
        on="worker_task_id",
        suffixes=("", "_hard_limit_min"),
    )
    return result_data.merge(
        neurite_hard_max,
        how="left",
        on="worker_task_id",
        suffixes=("", "_hard_limit_max"),
    )


def plot_scale_statistics(mtypes, scale_data, output_dir="scales", dpi=100):
    """Plot collage of an mtype and a list of planes.

    Args:
        mtypes (list): mtypes of cells to plot
        scale_data (dict): dicto od DataFrame(s) with scale data
        output_dir (str): result directory
        dpi (int): resolution of the output image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot statistics
    filename = Path(output_dir) / "statistics.pdf"
    with PdfPages(filename) as pdf:
        if scale_data.empty:
            fig = plt.figure(figsize=(10, 20))
            ax = plt.gca()
            ax.text(
                0.5,
                0.5,
                "NO DATA TO PLOT",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            with DisableLogger():
                pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

        if mtypes is None:
            mtypes = sorted(scale_data["mtype"].unique())

        for col in scale_data.drop(
            columns=["worker_task_id", "mtype", "x", "y", "z"], errors="ignore"
        ).columns:
            fig = plt.figure(figsize=(10, 20))
            ax = plt.gca()
            scale_data.loc[scale_data["mtype"].isin(mtypes), ["mtype", col]].boxplot(
                by="mtype", vert=False, ax=ax
            )

            ax.grid(True)
            fig.suptitle("")
            with DisableLogger():
                pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)


def mvs_score(data1, data2, percentile=10):
    """Get the MED - MVS score.

    The MED - MVS is equal to the absolute difference between the median of the
    population and the median of the neuron divided by the maximum visible spread.

    Args:
        data1 (list): the first data set.
        data2 (list): the second data set.
        percentile (int): percentile to compute.
    """
    median_diff = np.abs(np.median(data1) - np.median(data2))
    max_percentile = np.max(
        [
            np.percentile(data1, 100 - percentile / 2.0, axis=0),
            np.percentile(data2, 100 - percentile / 2.0, axis=0),
        ]
    )

    min_percentile = np.min(
        [
            np.percentile(data1, percentile / 2.0, axis=0),
            np.percentile(data2, percentile / 2.0, axis=0),
        ]
    )

    max_vis_spread = max_percentile - min_percentile

    return median_diff / max_vis_spread


def get_scores(df1, df2, percentile=5):
    """Return scores between two data sets.

    Args:
        df1 (pandas.DataFrame): the first data set.
        df2 (pandas.DataFrame): the second data set.
        percentile (int): percentile to compute.

    Returns:
        The list of feature scores.
    """
    scores = []
    score_names = []
    key_names = {
        "basal_dendrite": "Basal",
        "apical_dendrite": "Apical",
    }
    for neurite_type in ["basal_dendrite", "apical_dendrite"]:
        for k in df1.keys():
            if k in ["name", "neurite_type"]:
                continue
            data1 = df1.loc[df1["neurite_type"] == neurite_type, k]
            data2 = df2.loc[df2["neurite_type"] == neurite_type, k]
            sc1 = mvs_score(data1, data2, percentile)
            score_name = key_names[neurite_type] + " " + k.replace("_", " ")
            score_names.append(score_name)
            if sc1 is not np.nan:
                scores.append(sc1)
            else:
                scores.append(0.0)

    return score_names, scores


def compute_scores(ref, test, config):
    """Compute scores of a test population against a reference population.

    Args:
        ref (tuple(str, list)): the reference data.
        test (tuple(str, list)): the test data.
        config (dict): the configuration used to compute the scores.

    Returns:
        The scores and the feature list.
    """
    ref_mtype, ref_files = ref
    test_mtype, test_files = test
    assert ref_mtype == test_mtype, "The mtypes of ref and test files must be the same."

    ref_pop = load_neurons(ref_files)
    test_pop = load_neurons(test_files)

    ref_features = morph_stats.extract_dataframe(ref_pop, config)
    test_features = morph_stats.extract_dataframe(test_pop, config)

    return get_scores(ref_features, test_features, 5)


# pylint: disable=too-many-locals
def plot_score_matrix(
    ref_morphs_df,
    test_morphs_df,
    output_path,
    config,
    mtypes=None,
    path_col="filepath",
    dpi=100,
    nb_jobs=-1,
):
    """Plot score matrix for a test population against a reference population."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if mtypes is None:
        mtypes = sorted(ref_morphs_df.mtype.unique().tolist())

    def build_file_list(df, mtypes, path_col):
        return [
            (mtype, df.loc[df.mtype == mtype, path_col].sort_values().to_list()) for mtype in mtypes
        ]

    # Build the file list for each mtype
    ref_file_lists = build_file_list(ref_morphs_df, mtypes, path_col)
    test_file_lists = build_file_list(test_morphs_df, mtypes, path_col)

    # Compute scores
    scores = []
    keys = []
    for key_name, score in Parallel(nb_jobs)(
        delayed(compute_scores)(
            ref_files,
            test_files,
            config,
        )
        for ref_files, test_files in zip(ref_file_lists, test_file_lists)
    ):
        keys.append(key_name)
        scores.append(score)

    n_scores = len(keys[0])
    for k, s in zip(keys[1:], scores):
        assert keys[0] == k, "Score names must all be the same for each feature."
        assert len(k) == n_scores, "The number of keys must be the same for each mtype."
        assert len(s) == n_scores, "The number of scores must be the same for each mtype."

    # Plot statistics
    with PdfPages(output_path) as pdf:

        # Compute subplot ratios and figure size
        height_ratios = [7, (1 + n_scores)]
        fig_width = len(mtypes)
        fig_height = sum(height_ratios) * 0.3

        hspace = 0.625 / fig_height
        wspace = 0.2 / fig_width

        cbar_ratio = 0.4 / fig_width

        # Create the figure and the subplots
        fig, ((a0, a2), (a1, a3)) = plt.subplots(
            2,
            2,
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": [1 - cbar_ratio, cbar_ratio],
                "hspace": hspace,
                "wspace": wspace,
            },
            figsize=(fig_width, fig_height),
        )

        # Plot score errors
        a0.errorbar(
            np.arange(len(mtypes)),
            np.nanmean(scores, axis=1),
            yerr=np.nanstd(scores, axis=1),
            color="black",
            label="Synthesized",
        )
        a0.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
        a0.xaxis.set_tick_params(rotation=45)
        a0.set_xticks(np.arange(len(mtypes)))
        a0.set_xticklabels(mtypes)

        a0.set_xlim([a0.xaxis.get_ticklocs().min() - 0.5, a0.xaxis.get_ticklocs().max() + 0.5])
        a0.set_ylim([-0.1, 1.1])

        # Plot score heatmap
        scores_T = np.transpose(scores)
        scores_df = pd.DataFrame(scores_T, index=keys[0], columns=mtypes)

        g = sns.heatmap(
            scores_df,
            vmin=0,
            vmax=1,
            mask=np.isnan(scores_T),
            ax=a1,
            cmap=cm.Greys,  # pylint: disable=no-member
            cbar_ax=a3,
        )

        g.xaxis.set_tick_params(rotation=45)
        g.set_facecolor("xkcd:maroon")

        # Remove upper right subplot
        a2.remove()

        # Export the figure
        with DisableLogger():
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
