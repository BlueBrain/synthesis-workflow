"""Functions for validation of synthesis to be used by luigi tasks."""

# pylint: disable=too-many-lines
import itertools
import json
import logging
import multiprocessing
import os
import warnings
from collections import defaultdict
from collections import namedtuple
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import neurom
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel
from joblib import delayed
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.transform import transform
from morphio.mut import Morphology
from neurom import load_morphologies
from neurom.apps import morph_stats
from neurom.apps.morph_stats import extract_dataframe
from neurom.check.morphology_checks import has_apical_dendrite
from neurom.core.dataformat import COLS
from neurots import NeuronGrower
from neurots.extract_input.from_neurom import trunk_vectors
from neurots.generate.orientations import get_probability_function
from region_grower.modify import scale_target_barcode
from scipy.ndimage import correlate
from tmd.io.io import load_population
from voxcell import CellCollection

from synthesis_workflow.fit_utils import clean_outliers
from synthesis_workflow.fit_utils import get_path_distance_from_extent
from synthesis_workflow.fit_utils import get_path_distances
from synthesis_workflow.fit_utils import get_projections
from synthesis_workflow.tools import ensure_dir
from synthesis_workflow.utils import DisableLogger

L = logging.getLogger(__name__)
matplotlib.use("Agg")


VacuumCircuit = namedtuple("VacuumCircuit", ["cells", "morphs_df", "morphology_path"])
AtlasCircuit = namedtuple("AtlasCircuit", ["atlas", "cells", "morphology_path"])

SYNTH_MORPHOLOGY_PATH = "synth_morphology_path"


def convert_circuit_to_morphs_df(circuit_path, synth_output_path, ext="asc"):
    """Convert the list of morphologies from circuit to morphology dataframe.

    Args:
        circuit_path (str): path to circuit file
        synth_output_path (str): path to morphology files
        ext( str): extension of morphology files

    Returns:
        DataFrame: morphology dataframe
    """
    cells_df = CellCollection.load(circuit_path).as_dataframe()
    cells_df[SYNTH_MORPHOLOGY_PATH] = cells_df["morphology"].apply(
        lambda morph: (Path(synth_output_path) / morph).with_suffix("." + ext)
    )
    cells_df["name"] = cells_df["morphology"]
    return cells_df.drop("morphology", axis=1)


def get_features_df(morphologies_mtypes: Dict, features_config: Dict, n_workers: int = 1):
    """Create a feature dataframe from a dictionary of morphology_folders per mtypes.

    Args:
        morphologies_mtypes (dict): dict of morphology_folders files per mtype
        features_config (dict): configuration dict for features extraction
            (see ``neurom.apps.morph_stats.extract_dataframe``)
        n_workers (int) : number of workers for feature extractions
    """
    if n_workers == -1:
        n_workers = multiprocessing.cpu_count()
    rows = []
    for mtype, morphology_folders in morphologies_mtypes.items():
        features_df_tmp = extract_dataframe(
            morphology_folders, features_config, n_workers=n_workers
        )
        features_df_tmp["mtype"] = mtype
        rows.append(features_df_tmp.replace(0, np.nan))
    features_df = pd.concat(rows)
    return features_df


def _get_features_df_all_mtypes(morphs_df, features_config, morphology_path, n_workers=None):
    """Wrapper for morph-validator functions."""
    if n_workers is None:
        n_workers = os.cpu_count()
    morphs_df_dict = {mtype: df[morphology_path] for mtype, df in morphs_df.groupby("mtype")}
    with warnings.catch_warnings():
        # Ignore some Numpy warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return get_features_df(morphs_df_dict, features_config, n_workers=n_workers)


def get_feature_configs(config_types="default"):
    """Getter function of default features configs.

    Currently available config_types:
        - default
        - repair
        - synthesis

    Args:
        config_types (list/str): list of types of config files

    """
    CONFIG_DEFAULT = {
        "neurite": {"total_length_per_neurite": ["sum"]},
    }

    CONFIG_REPAIR = {
        "neurite": {"total_length_per_neurite": ["sum"]},
    }

    CONFIG_SYNTHESIS = {
        "neurite": {
            "number_of_neurites": ["sum"],
            "number_of_sections_per_neurite": ["mean"],
            "number_of_leaves": ["sum"],
            "number_of_bifurcations": ["sum"],
            "section_lengths": ["mean"],
            "section_radial_distances": ["mean"],
            "section_path_distances": ["mean"],
            "section_branch_orders": ["mean"],
            "remote_bifurcation_angles": ["mean"],
        },
        "neurite_type": ["BASAL_DENDRITE", "APICAL_DENDRITE"],
    }

    if not isinstance(config_types, list):
        config_types = [config_types]
    features_config = {}
    for config_type in config_types:
        if config_type == "default":
            features_config.update(CONFIG_DEFAULT)

        if config_type == "repair":
            features_config.update(CONFIG_REPAIR)

        if config_type == "synthesis":
            features_config.update(CONFIG_SYNTHESIS)

    if not features_config:
        raise ValueError("No features_config could be created with " + str(config_types))
    return features_config


def _expand_lists(data):
    """Convert list element of dataframe to duplicated rows with float values."""
    new_rows = []
    for row_id in data.index:
        if isinstance(data.loc[row_id, "value"], list):
            for value in data.loc[row_id, "value"]:
                new_row = data.loc[row_id].copy()
                new_row["value"] = value
                new_rows.append(new_row)
        else:
            new_rows.append(data.loc[row_id].copy())
    data_expanded = pd.DataFrame(new_rows)
    return data_expanded


def _normalize(data):
    """Normalize data with mean and std."""
    if len(data) == 0:
        return data
    data_tmp = data.set_index(["feature", "neurite_type", "mtype"])
    groups = data_tmp.groupby(["feature", "neurite_type", "mtype"])
    means = groups.mean().reset_index()
    stds = groups.std().reset_index()
    for feat_id in means.index:
        mask = (
            (data.feature == means.loc[feat_id, "feature"])
            & (data.neurite_type == means.loc[feat_id, "neurite_type"])
            & (data.mtype == means.loc[feat_id, "mtype"])
        )
        data.loc[mask, "value"] = (
            data.loc[mask, "value"] - means.loc[feat_id, "value"]
        ) / stds.loc[feat_id, "value"]
    return data


def plot_violin_features(
    features: pd.DataFrame, neurite_types: List, output_dir: Path, bw: float, normalize=True
):
    """Create violin plots from features dataframe.

    Args:
        features (pandas.DataFrame): features dataframe to plot
        neurite_types (list): list of neurite types to plot (one plot per neurite_type)
        output_dir (Path): path to folder for saving plots
        bw (float): resolution of violins
        normalize (bool): normalize feature values with mean/std
    """
    L.info("Plotting features.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if "property" in features.columns:
        features = features.drop(columns="property")
    features = features.melt(id_vars=[("mtype", ""), ("label", "")]).rename(
        columns={
            ("mtype", ""): "mtype",
            ("label", ""): "label",
            "variable_0": "neurite_type",
            "variable_1": "feature",
        }
    )
    for neurite_type in neurite_types:
        with PdfPages(output_dir / f"morphometrics_{neurite_type}.pdf") as pdf:
            for mtype in features.mtype.unique():
                data = features.loc[
                    (features.mtype == mtype) & (features.neurite_type == neurite_type)
                ].dropna()
                data = _expand_lists(data)
                if normalize:
                    data = _normalize(data)

                if len(data.index) > 0:
                    plt.figure()
                    ax = plt.gca()
                    sns.violinplot(
                        x="feature",
                        y="value",
                        hue="label",
                        data=data,
                        split=True,
                        bw=bw,
                        ax=ax,
                        inner="quartile",
                    )

                    ax.tick_params(axis="x", rotation=90)
                    plt.suptitle(f"mtype: {mtype}")
                    pdf.savefig(bbox_inches="tight")
                    plt.close()


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
    n_workers=None,
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
        n_workers (int): the number of workers used to compute morphology features
    """
    if config_features is None:
        config_features = get_feature_configs(config_types="synthesis")
        # config_features["neurite"].update({"y_distances": ["min", "max"]})

    L.debug("Get features from base morphologies")
    base_features_df = _get_features_df_all_mtypes(
        base_morphs_df, config_features, base_key, n_workers=n_workers
    )
    base_features_df["label"] = base_label

    L.debug("Get features from compared morphologies")
    comp_features_df = _get_features_df_all_mtypes(
        comp_morphs_df, config_features, comp_key, n_workers=n_workers
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
    with DisableLogger():
        L.debug("Plot violin figure to %s", str(output_path))
        plot_violin_features(
            all_features_df,
            ["basal_dendrite", "apical_dendrite"],
            output_dir=Path(output_path),
            bw=0.1,
            normalize=normalize,
        )


def iter_positions(morph, sample_distance, neurite_filter=None):
    """Iterator for positions in space of points every <sample_distance> um.

    Assumption about segment linearity is that curvature between the start and end of segments
    is negligible.

    Args:
        morph (neurom.FstNeuron): morphology
        sample_distance (int): points sampling distance (in um)
        neurite_filter: filter neurites, see ``neurite_filter`` of ``neurom.iter_sections()``

    Yields:
        sampled points for the neurites (each point is a (3,) numpy array).
    """
    section_offsets = {}

    for section in neurom.iter_sections(morph, neurite_filter=neurite_filter):
        if section.parent is None:
            parent_section_offset = 0
        else:
            parent_section_offset = section_offsets[section.parent.id]
        segment_offset = parent_section_offset
        for segment in neurom.iter_segments(section):
            segment_len = neurom.morphmath.segment_length(segment)
            if segment_offset + segment_len < sample_distance:
                segment_offset += segment_len
            elif segment_offset + segment_len == sample_distance:
                yield segment[1][COLS.XYZ]
                segment_offset = 0
            else:
                offsets = np.arange(sample_distance - segment_offset, segment_len, sample_distance)
                for offset in offsets:
                    yield neurom.morphmath.linear_interpolate(*segment, offset / segment_len)
                segment_offset = segment_len - offsets[-1]
                if segment_offset == sample_distance:
                    segment_offset = 0
                    yield segment[1][COLS.XYZ]
        section_offsets[section.id] = segment_offset


def sample_morph_voxel_values(
    morphology,
    sample_distance,
    voxel_data,
    out_of_bounds_value,
    neurite_types=None,
):
    """Sample the values of the neurites in the given voxeldata.

    The value is out_of_bounds_value if the neurite is outside the voxeldata.

    Arguments:
        morphology (neurom.FstNeuron): cell morphology
        sample_distance (int in um): sampling distance for neurite points
        voxel_data (voxcell.VoxelData): volumetric data to extract values from
        out_of_bounds_value: value to assign to neurites outside of voxeldata
        neurite_types (list): list of neurite types, or None (will use basal and axon)

    Returns:
        dict mapping each neurite type of the morphology to the sampled values
        {(neurom.NeuriteType): np.array(...)}
    """
    if neurite_types is None:
        neurite_types = [neurite.type for neurite in morphology.neurites]

    output = {}
    for neurite_type in neurite_types:
        points = list(
            iter_positions(
                morphology,
                sample_distance=sample_distance,
                neurite_filter=lambda n, nt=neurite_type: n.type == nt,
            )
        )
        indices = voxel_data.positions_to_indices(points, False)
        out_of_bounds = np.any(indices == -1, axis=1)
        within_bounds = ~out_of_bounds
        values = np.zeros(len(points), dtype=voxel_data.raw.dtype)
        values[within_bounds] = voxel_data.raw[tuple(indices[within_bounds].transpose())]
        values[out_of_bounds] = out_of_bounds_value
        output[neurite_type] = values
    return output


def _get_depths_df(circuit, mtype, sample, voxeldata, sample_distance):
    """Create dataframe with depths data for violin plots."""
    out_of_bounds_value = np.nan
    morphs_df = circuit.morphs_df
    path = Path(circuit.morphology_path)
    cells = morphs_df.loc[morphs_df["mtype"] == mtype, path]
    gids = cells.sample(sample, random_state=42).index

    point_depths = defaultdict(list)
    for gid in gids:
        morphology = Morphology(path / (cells.loc[gid, "morph"] + ".asc"))
        T = np.eye(4)
        T[:3, :3] = cells.loc[gid, "orientation"]
        T[:3, 3] = cells.loc[gid, ["x", "y", "z"]]
        transform(morphology, T)
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
            label=f"layer {layer + 1}",
        )


def _plot_density_profile(
    mtype, circuit=None, x_pos=None, sample=None, voxeldata=None, sample_distance=None
):
    """Plot density profile of an mtype."""
    fig = plt.figure()
    ax = plt.gca()
    try:
        if isinstance(circuit, AtlasCircuit):
            _plot_layers(x_pos, circuit.atlas, ax)
            plot_df = _get_depths_df(circuit, mtype, sample, voxeldata, sample_distance)
            ax.legend(loc="best")
        elif isinstance(circuit, VacuumCircuit):
            plot_df = _get_vacuum_depths_df(circuit, mtype)
        else:
            plot_df = None

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


def _spherical_filter(radius):
    filt_size = radius * 2 + 1
    sphere = np.zeros((filt_size, filt_size, filt_size))
    center = np.array([radius, radius, radius])
    posns = np.transpose(np.nonzero(sphere == 0))
    in_sphere = posns[np.linalg.norm(posns - center, axis=-1) <= radius]
    sphere[tuple(in_sphere.transpose())] = 1
    return sphere


def relative_depth_volume(
    atlas,
    top_layer="1",
    bottom_layer="6",  # pylint: disable=too-many-locals
    in_region="Isocortex",
    relative=True,
):
    """Return volumetric data of relative cortical depth at voxel centers.

    The relative cortical depth is equal to <distance from pia> / <total_cortex_thickness>.
    Outside of the region `in_region` relative depth will be estimated, i.e. extrapolated from the
    internal relative depth.
    The `in_region` is the region within which to use the relative depth-values outside this
    region are estimated.
    """
    y = atlas.load_data("[PH]y")
    top = atlas.load_data(f"[PH]{top_layer}").raw[..., 1]
    bottom = atlas.load_data(f"[PH]{bottom_layer}").raw[..., 0]
    thickness = top - bottom
    if relative:
        reldepth = (top - y.raw) / thickness
    else:
        reldepth = y.raw
    voxel_size = y.voxel_dimensions[0]
    if in_region is None:
        raise ValueError("in_region should not be None")
    region = atlas.get_region_mask(in_region).raw
    to_filter = np.zeros(region.shape)
    to_filter[np.logical_and(region, reldepth < 0.5)] = -1
    to_filter[np.logical_and(region, reldepth >= 0.5)] = 1
    max_dist = 5  # voxels
    for voxels_distance in range(max_dist, 0, -1):
        filt = _spherical_filter(voxels_distance)

        num_voxels_in_range = correlate(to_filter, filt)
        # we get the estimated thickness by normalizing the filtered thickness
        # by the number of voxels that contributed
        filtered_thickness = correlate(region * np.nan_to_num(thickness), filt) / np.abs(
            num_voxels_in_range
        )
        in_range_below = np.logical_and(num_voxels_in_range > 0.5, ~region)
        in_range_above = np.logical_and(num_voxels_in_range < -0.5, ~region)
        max_distance_from_region = voxels_distance * voxel_size
        reldepth[in_range_below] = 1 + (
            max_distance_from_region / filtered_thickness[in_range_below]
        )
        reldepth[in_range_above] = -(max_distance_from_region / filtered_thickness[in_range_above])
    return y.with_data(reldepth)


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


def _generate_synthetic_random_population(
    dir_path, nb, proj_min, proj_max, tmd_parameters, tmd_distributions
):
    """Generate a synthetic population with random projections."""
    files = []
    y_synth = []
    slope = tmd_parameters["context_constraints"]["apical_dendrite"]["extent_to_target"]["slope"]
    intercept = tmd_parameters["context_constraints"]["apical_dendrite"]["extent_to_target"][
        "intercept"
    ]
    for i in range(nb):
        tmp_name = str((Path(dir_path) / str(i)).with_suffix(".h5"))
        files.append(tmp_name)
        projection = np.random.randint(proj_min, proj_max)
        y_synth.append(projection)
        target_path_distance = get_path_distance_from_extent(slope, intercept, projection)
        tmd_parameters["apical_dendrite"].update(
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
    mtype,
    files,
    outlier_percentage,
    tmd_parameters,
    tmd_distributions,
    neurite_type="apical_dendrite",
):
    """Get projections and path lengths of a given and a synthetic population."""
    # Load biological neurons
    return_error = (mtype, None, None, None, None, None, None)
    if len(files) > 0:
        input_population = load_population(files)
    else:
        return return_error + (f"No file to load for mtype='{mtype}'",)
    if (
        tmd_parameters.get("context_constraints", {})
        .get("apical_dendrite", {})
        .get("extent_to_target")
        is None
    ):
        return return_error + (f"No fit for mtype='{mtype}'",)

    # Get X and Y from biological population
    x = get_path_distances(input_population, neurite_type)
    y = get_projections(input_population, neurite_type)
    x_clean, y_clean = clean_outliers(x, y, outlier_percentage)

    # Create synthetic neuron population
    tmd_distributions["diameter"] = {"method": "M1"}
    tmd_parameters["diameter_params"] = {"method": "M1"}

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
    region=None,
    outlier_percentage=90,
    nb_jobs=-1,
    neurite_type="apical_dendrite",
):
    """Plot path-distance fits."""
    # Read TMD parameters
    with open(tmd_parameters_path, "r", encoding="utf-8") as f:
        tmd_parameters = json.load(f)

    # Read TMD distributions
    with open(tmd_distributions_path, "r", encoding="utf-8") as f:
        tmd_distributions = json.load(f)

    # Read morphology DataFrame
    morphs_df = pd.read_csv(morphs_df_path)

    if mtypes is None:
        mtypes = sorted(
            [
                mtype
                for mtype in morphs_df.mtype.unique().tolist()
                if tmd_parameters[region]
                .get(mtype, {})
                .get("context_constraints", {})
                .get("apical_dendrite", {})
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
                tmd_parameters[region][mtype],
                tmd_distributions[region][mtype],
                neurite_type=neurite_type,
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
            plt.scatter(x_synth, y_synth, c="green", label="Synthesized points")

            try:
                # Plot fit function
                plt.plot(
                    [
                        get_path_distance_from_extent(
                            tmd_parameters[region][mtype]["context_constraints"]["apical_dendrite"][
                                "extent_to_target"
                            ]["slope"],
                            tmd_parameters[region][mtype]["context_constraints"]["apical_dendrite"][
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


def get_debug_data(log_file):
    """Parse log file and return a DataFrame with data."""
    # Get data
    df = pd.read_pickle(log_file)

    # Compute min/max hard limit scales
    hard_scales = df["debug_infos"].apply(
        lambda row: [i["scale"] for i in row.get("neurite_hard_limit_rescaling", {}).values()]
    )
    neurite_hard_min = hard_scales.apply(lambda row: min(row) if row else None)
    neurite_hard_max = hard_scales.apply(lambda row: max(row) if row else None)

    default_scales = df["debug_infos"].apply(
        lambda row: [
            i["scaling_ratio"]
            for i in row.get("input_scaling", {}).get("default_func", {}).get("scaling", {})
        ]
    )
    default_min = default_scales.apply(lambda row: min(row) if row else None)
    default_max = default_scales.apply(lambda row: max(row) if row else None)

    target_scales = df["debug_infos"].apply(
        lambda row: [
            i["scaling_ratio"]
            for i in row.get("input_scaling", {}).get("target_func", {}).get("scaling", {})
        ]
    )
    target_min = target_scales.apply(lambda row: min(row) if row else None)
    target_max = target_scales.apply(lambda row: max(row) if row else None)

    # Merge data
    stat_cols = {
        "min_hard_scale": neurite_hard_min,
        "max_hard_scale": neurite_hard_max,
        "min_default_scale": default_min,
        "max_default_scale": default_max,
        "min_target_scale": target_min,
        "max_target_scale": target_max,
    }

    for col, vals in stat_cols.items():
        df[col] = vals

    return df, list(stat_cols.keys())


def plot_scale_statistics(mtypes, scale_data, cols, output_dir="scales", dpi=100):
    """Plot collage of an mtype and a list of planes.

    Args:
        mtypes (list): mtypes of cells to plot
        scale_data (pd.DataFrame): DataFrame with scale data
        cols (list): The column names that should be plotted from the data
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

        for col in cols:
            fig = plt.figure()
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
        if neurite_type in df1.columns and neurite_type in df2.columns:
            _df1 = df1[neurite_type]
            _df2 = df2[neurite_type]
            for k in _df1.columns:
                data1 = _df1[k]
                data2 = _df2[k]
                data1 = data1[~data1.isna()]
                data2 = data2[~data2.isna()]
                score_name = key_names[neurite_type] + " " + k.replace("_", " ")
                score_names.append(score_name)
                if len(data1.index) > 0 and len(data2.index) > 0:
                    sc1 = mvs_score(data1, data2, percentile)
                    if sc1 is not np.nan:
                        scores.append(sc1)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(np.nan)

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

    ref_pop = load_morphologies(ref_files)
    test_pop = load_morphologies(test_files)

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


def extract_angle_data(df, morph_key, pia=None):
    """Extract all pairwise angles between neurite_types from the morphologies in df."""
    if pia is None:
        pia = [0, 1, 0]

    data = defaultdict(list)
    morph_class = (
        "PC" if has_apical_dendrite(neurom.load_morphology(df.iloc[0][morph_key])) else "IN"
    )

    for gid in df.index:
        morph = neurom.load_morphology(df.loc[gid, morph_key])

        _vec_basal = trunk_vectors(morph, neurite_type=neurom.NeuriteType.basal_dendrite)
        _vec_axon = trunk_vectors(morph, neurite_type=neurom.NeuriteType.axon)

        if morph_class == "PC":
            _vec_apical = trunk_vectors(morph, neurite_type=neurom.NeuriteType.apical_dendrite)
            data["pia__apical_dendrite"] += [
                neurom.morphmath.angle_between_vectors(_vec_apical[0], [0, 1, 0])
            ]

        data["basal_dendrite__basal_dendrite"] += [
            neurom.morphmath.angle_between_vectors(_vec_basal[i], _vec_basal[j])
            for i, j in itertools.combinations(range(len(_vec_basal)), 2)
        ]

        if morph_class == "PC":
            data["apical_dendrite__basal_dendrite"] += [
                neurom.morphmath.angle_between_vectors(_vec_apical[0], _vec_basal[i])
                for i in range(len(_vec_basal))
            ]

        data["pia__basal_dendrite"] += [
            neurom.morphmath.angle_between_vectors(pia, _vec_basal[i])
            for i in range(len(_vec_basal))
        ]

        if len(_vec_axon) > 0:
            if morph_class == "PC":
                data["axon__apical_dendrite"] += [
                    neurom.morphmath.angle_between_vectors(_vec_axon[0], _vec_apical[0])
                ]
            data["axon__pia"] += [neurom.morphmath.angle_between_vectors(_vec_axon[0], [0, 1, 0])]
            data["axon__basal_dendrite"] += [
                neurom.morphmath.angle_between_vectors(_vec_axon[0], _vec_basal[i])
                for i in range(len(_vec_basal))
            ]
    return data


def _get_hist(data, bins=50):
    """Return density histogram with bin centers."""
    d, b = np.histogram(data, bins=bins, density=True)
    b = 0.5 * (b[1:] + b[:-1])
    return b, d / max(d)


def trunk_validation(
    morphs_df,
    synth_morphs_df,
    output_dir,
    base_key,
    comp_key,
    base_label,
    comp_label,
    tmd_parameters_path,
    tmd_distributions_path,
    region,
):
    """Create plots to validate trunk angles."""
    with open(tmd_parameters_path, "r", encoding="utf-8") as f:
        tmd_parameters = json.load(f)

    with open(tmd_distributions_path, "r", encoding="utf-8") as f:
        tmd_distributions = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    for mtype in morphs_df.mtype.unique():
        data_bio = extract_angle_data(morphs_df[morphs_df.mtype == mtype], base_key)
        data_synth = extract_angle_data(synth_morphs_df[synth_morphs_df.mtype == mtype], comp_key)
        with PdfPages(output_dir / f"trunk_validation_{mtype}.pdf") as pdf:
            for data_type in data_bio:
                t1, t2 = data_type.split("__")

                angles = np.linspace(0, np.pi, 100)
                fit = None
                if (
                    "params"
                    in tmd_parameters[region][mtype]
                    .get(t2, {})
                    .get("orientation", {})
                    .get("values", {})
                    and tmd_parameters[region][mtype][t2]["orientation"]["mode"].split("_")[0]
                    == t1.split("_")[0]
                ):
                    fit = get_probability_function(
                        form=tmd_parameters[region][mtype][t2]["orientation"]["values"]["form"],
                        with_density=True,
                    )(
                        angles,
                        *tmd_parameters[region][mtype][t2]["orientation"]["values"]["params"],
                    )
                plt.figure(figsize=(6, 4))
                plt.axvline(0, c="k")
                plt.axvline(np.pi, c="k")
                plt.plot(*_get_hist(data_bio[data_type]), label=base_label)
                key = t1.split("_")[0] + "_3d_angles"
                if key in tmd_distributions[region][mtype].get(t2, {}).get("trunk", {}):
                    bio_data = tmd_distributions[region][mtype][t2]["trunk"][key]["data"]
                    plt.plot(
                        bio_data["bins"],
                        np.array(bio_data["weights"]) / max(bio_data["weights"]),
                        label=base_label + "_data",
                    )

                if data_type in data_synth:
                    plt.plot(*_get_hist(data_synth[data_type]), label=comp_label)

                if fit is not None:
                    plt.plot(angles, fit, label="fit")
                plt.xlabel(data_type)
                plt.legend()
                pdf.savefig(bbox_inches="tight")
                plt.close()
