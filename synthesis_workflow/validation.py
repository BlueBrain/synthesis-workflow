"""Functions for validation of synthesis to be used by luigi tasks."""
import json
import glob
import logging
import os
import re
import warnings
from collections import defaultdict
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
from matplotlib.backends.backend_pdf import PdfPages
from pyquaternion import Quaternion
from scipy.optimize import fmin
from tqdm import tqdm

from atlas_analysis.constants import CANONICAL
from morph_validator.feature_configs import get_feature_configs
from morph_validator.plotting import get_features_df
from morph_validator.plotting import plot_violin_features
from morph_validator.spatial import relative_depth_volume
from morph_validator.spatial import sample_morph_voxel_values
from neurom import viewer
from region_grower.atlas_helper import AtlasHelper
from region_grower.modify import scale_target_barcode
from tmd.io.io import load_population
from tns import NeuronGrower
from voxcell import CellCollection
from voxcell.exceptions import VoxcellError

from .circuit import get_cells_between_planes
from .fit_utils import clean_outliers
from .fit_utils import get_path_distances
from .fit_utils import get_path_distance_from_extent
from .fit_utils import get_projections
from .tools import ensure_dir
from .utils import DisableLogger


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
        atlas (AtlasHelper): if atlas is provided, we will plot arrows with orientations
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


def get_y_info(atlas, plane_origin, rotation_matrix, n_pixels=64):
    """Get direction of y axis on a grid on the atlas planes."""
    bbox = atlas.orientations.bbox
    bbox_min = rotation_matrix.dot(bbox[0] - plane_origin)
    bbox_max = rotation_matrix.dot(bbox[1] - plane_origin)
    xs_plane = np.linspace(bbox_min[0], bbox_max[0], n_pixels)
    ys_plane = np.linspace(bbox_min[1], bbox_max[1], n_pixels)

    orientation_u = np.zeros([n_pixels, n_pixels])
    orientation_v = np.zeros([n_pixels, n_pixels])
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
            try:
                orientation = atlas.lookup_orientation(point)
                if orientation[0] != 0.0 and orientation[1] != 1.0:
                    orientation_u[i, j], orientation_v[i, j] = rotation_matrix.dot(
                        orientation
                    )[:2]
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

        def _to_plane_coord(p):
            return np.dot(p - plane_left.point, rotation_matrix.T)

        # transform morphology in the plane coordinates
        morphology = morphology.transform(_to_plane_coord)

        if atlas is not None:
            pos_orig = circuit.cells.positions(gid).to_numpy()
            pos_final = pos_orig + atlas.lookup_orientation(pos_orig) * 300
            pos_orig = _to_plane_coord(pos_orig)
            pos_final = _to_plane_coord(pos_final)
            plt.plot(
                [pos_orig[0], pos_final[0]],
                [pos_orig[1], pos_final[1]],
                c="0.5",
                lw=0.2,
            )

        viewer.plot_neuron(
            ax, morphology, plane="xy", realistic_diameters=True, linewidth=0.1
        )


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
):
    """Internal plot collage for multiprocessing."""
    if with_y_field and atlas is None:
        raise Exception("Please provide an atlas with option with_y_field=True")

    left_plane, right_plane = planes
    rotation_matrix = get_plane_rotation_matrix(left_plane)
    X, Y, layers = get_layer_info(
        layer_annotation, left_plane.point, rotation_matrix, n_pixels
    )
    if with_y_field:
        X_y, Y_y, orientation_u, orientation_v = get_y_info(
            atlas, left_plane.point, rotation_matrix, n_pixels_y
        )

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
        )
    if with_y_field:
        # note: some of these parameters are harcoded for NCx plot, adjust as needed
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
        n_pixels (int): number of pixels for plotting layers
        with_y_field (bool): plot y field
        n_pixels_y (int): number of pixels for plotting y field
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
    """Generate a synthetic population with random projections"""
    files = []
    y_synth = []
    slope = tmd_parameters["context_constraints"]["apical"]["extent_to_target"]["slope"]
    intercept = tmd_parameters["context_constraints"]["apical"]["extent_to_target"][
        "intercept"
    ]
    for i in range(nb):
        tmp_name = str((Path(dir_path) / str(i)).with_suffix(".h5"))
        files.append(tmp_name)
        projection = np.random.randint(proj_min, proj_max)
        y_synth.append(projection)
        target_path_distance = get_path_distance_from_extent(
            slope, intercept, projection
        )
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


def _get_fit_population(
    mtype, files, outlier_percentage, tmd_parameters, tmd_distributions
):
    """Get projections and path lengths of a given biological population and a
    synthetic population based on the first one."""

    # Load biological neurons
    if len(files) > 0:
        input_population = load_population(files)
    else:
        return (mtype, None, None)

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

    return mtype, x, y, x_clean, y_clean, x_synth, y_synth


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
    """Plot path-distance fits"""

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

    ensure_dir(output_path)
    with PdfPages(output_path) as pdf:
        for mtype, x, y, x_clean, y_clean, x_synth, y_synth in Parallel(nb_jobs)(
            delayed(_get_fit_population)(
                mtype,
                files,
                outlier_percentage,
                tmd_parameters[mtype],
                tmd_distributions["mtypes"][mtype],
            )
            for mtype, files in file_lists
        ):
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
    """Parse log file and return a DataFrame with data"""
    # pylint: disable=too-many-locals
    # TODO: update this when region-grower is ready

    def _search(pattern, line, data):
        group = re.search(pattern, line)
        if group:
            groups = group.groups()
            new_data = json.loads(groups[1])
            new_data["worker_task_id"] = groups[0]
            data.append(new_data)

    # List log files
    log_files = glob.glob(log_file + "*")

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
        df[["x", "y", "z"]] = pd.DataFrame(
            df["position"].values.tolist(), index=df.index
        )
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
    result_data = pd.merge(
        result_data, default_min, on="worker_task_id", suffixes=("", "_default_min")
    )
    result_data = pd.merge(
        result_data, default_max, on="worker_task_id", suffixes=("", "_default_max")
    )
    result_data = pd.merge(
        result_data, target_min, on="worker_task_id", suffixes=("", "_target_min")
    )
    result_data = pd.merge(
        result_data, target_max, on="worker_task_id", suffixes=("", "_target_max")
    )
    result_data = pd.merge(
        result_data,
        neurite_hard_min,
        on="worker_task_id",
        suffixes=("", "_hard_limit_min"),
    )
    result_data = pd.merge(
        result_data,
        neurite_hard_max,
        on="worker_task_id",
        suffixes=("", "_hard_limit_max"),
    )

    return result_data


def plot_scale_statistics(mtypes, scale_data, output_dir="scales", dpi=100):
    """Plot collage of an mtype and a list of planes.

    Args:
        mtypes (list): mtypes of cells to plot
        scale data (dict): dicto od DataFrame(s) with scale data
        output_dir (str): result directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot statistics
    filename = Path(output_dir) / "statistics.pdf"
    with PdfPages(filename) as pdf:
        if scale_data.empty:
            fig = plt.figure()
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
            mtypes = scale_data["mtype"].unique().tolist()
        for col in scale_data.columns:
            if col in ["worker_task_id", "mtype", "x", "y", "z"]:
                continue
            ax = scale_data[["mtype", col]].boxplot(by="mtype")

            fig = ax.figure
            ax.grid(True)
            fig.suptitle("")
            with DisableLogger():
                pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
