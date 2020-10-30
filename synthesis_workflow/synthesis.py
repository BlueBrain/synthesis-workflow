"""Functions for synthesis to be used by luigi tasks."""
import logging
import os
import re
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import cpu_count
from tqdm import tqdm

from morphio.mut import Morphology
from neuroc.scale import scale_section
from neuroc.scale import ScaleParameters
from neurom.core.dataformat import COLS
from placement_algorithm import files
from placement_algorithm.app import utils
from placement_algorithm.app import choose_morphologies
from placement_algorithm.app.choose_morphologies import Worker as ChooseMorphologyWorker
from placement_algorithm.app.synthesize_morphologies import (
    Master as SynthesizeMorphologiesMaster,
)
from tmd.io.io import load_population
from tns import extract_input
from voxcell import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from . import STR_TO_TYPES
from .fit_utils import fit_path_distance_to_extent
from .tools import run_master


L = logging.getLogger(__name__)
matplotlib.use("Agg")


def get_neurite_types(morphs_df, mtypes):
    """Get the neurite types to consider for PC or IN cells."""
    return {
        mtype: ["basal"]
        if morphs_df.loc[morphs_df.mtype == mtype, "morph_class"].tolist()[0] == "IN"
        else ["basal", "apical"]
        for mtype in mtypes
    }


def apply_substitutions(original_morphs_df, substitution_rules=None):
    """apply substitution rule on .dat file.

    Args:
        original_morphs_df (DataFrame): dataframe with morphologies
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies

    Returns:
        DataFrame: dataframe with original and new morphologies
    """
    if not substitution_rules:
        return original_morphs_df

    new_morphs_df = original_morphs_df.copy()
    for gid in original_morphs_df.index:
        mtype_orig = original_morphs_df.loc[gid, "mtype"]
        if mtype_orig in substitution_rules:
            for mtype in substitution_rules[mtype_orig]:
                new_cell = original_morphs_df.loc[gid].copy()
                new_cell["mtype"] = mtype
                new_morphs_df = new_morphs_df.append(new_cell)
    return new_morphs_df


def _build_distributions_single_mtype(
    mtype,
    morphs_df=None,
    neurite_types=None,
    diameter_model_function=None,
    morphology_path=None,
):
    """Internal function for multiprocessing of tmd_distribution building."""
    morphology_paths = morphs_df.loc[
        morphs_df.mtype == mtype, morphology_path
    ].to_list()
    return (
        mtype,
        extract_input.distributions(
            morphology_paths,
            neurite_types=neurite_types[mtype],
            diameter_input_morph=morphology_paths,
            diameter_model=diameter_model_function,
        ),
    )


def build_distributions(
    mtypes,
    morphs_df,
    neurite_types,
    diameter_model_function,
    morphology_path,
    cortical_thickness,
    nb_jobs=-1,
    joblib_verbose=10,
):
    """Build tmd_distribution dictionary for synthesis.

    Args:
        mtypes (list): list of mtypes to build distribution for
        morphs_df (DataFrame): morphology dataframe with reconstruction to use
        diameter_model_function (function): diametrizer function (from diameter-synthesis)
        morphology_path (str): name of the column in morphs_df to use for paths to morphologies
        cortical_thickness (list): list of cortical thicknesses for reference scaling

    Returns:
        dict: dict to save to tmd_distribution.json
    """
    build_distributions_single_mtype = partial(
        _build_distributions_single_mtype,
        morphs_df=morphs_df,
        neurite_types=neurite_types,
        diameter_model_function=diameter_model_function,
        morphology_path=morphology_path,
    )

    tmd_distributions = {
        "mtypes": {},
        "metadata": {"cortical_thickness": cortical_thickness},
    }
    for mtype, distribution in Parallel(nb_jobs, verbose=joblib_verbose)(
        delayed(build_distributions_single_mtype)(mtype) for mtype in mtypes
    ):
        tmd_distributions["mtypes"][mtype] = distribution
    return tmd_distributions


def get_axon_base_dir(morphs_df, col_name="morphology_path"):
    """Get the common base directory of morphologies"""
    if morphs_df.empty:
        raise RuntimeError("Can not get axon base dir from an empty DataFrame")

    col = morphs_df[col_name]

    # Get all parent directories
    parents = col.apply(lambda x: str(Path(x).parent)).unique()

    # Check they are all equal
    if len(parents) != 1:
        raise Exception("Base dirs are different for axon grafting.")

    # Pick the first one
    axon_morphs_base_dir = parents[0]

    # Remove '-asc' suffix if present
    if axon_morphs_base_dir.split("-")[-1] == "asc":
        axon_morphs_base_dir = "-".join(axon_morphs_base_dir.split("-")[:-1])

    return axon_morphs_base_dir


# pylint: disable=too-many-arguments, too-many-locals
def create_axon_morphologies_tsv(
    circuit_path,
    morphs_df_path,
    atlas_path=None,
    annotations_path=None,
    rules_path=None,
    morphdb_path=None,
    alpha=1.0,
    scales=None,
    seed=0,
    axon_morphs_path="axon_morphs.tsv",
    nb_jobs=-1,
    joblib_verbose=10,
):
    """Create required axon_morphology tsv file for placement-algorithm to graft axons.

    Args:
        circuit_path (str): path to circuit somata file
        morphs_df_path (str): path to morphology dataframe
        axon_morphs_path (str): name of the axon morphology list in .tsv format

    Returns:
        str: path to base directory of morphologies for axon grafting
    """
    check_placement_params = [
        atlas_path is not None,
        annotations_path is not None,
        rules_path is not None,
    ]
    if any(check_placement_params) and not all(check_placement_params):
        raise ValueError(
            "Either 0 or all the following parameter should be None: %s"
            % [
                "atlas_path",
                "annotations_path",
                "rules_path",
            ]
        )

    if all(check_placement_params):
        L.info("Use placement algorithm for axons")

        atlas = Atlas.open(atlas_path)
        segment_type = "axon"

        rules = files.PlacementRules(rules_path)
        annotations = (
            choose_morphologies._bind_annotations(  # pylint: disable=protected-access
                utils.load_json(annotations_path),
                files.parse_morphdb(morphdb_path),
                rules,
            )
        )

        # TODO: use the Master class with tools.run_master() instead of the Worker?
        worker = ChooseMorphologyWorker(annotations, rules.layer_names, False)
        worker.cells = CellCollection.load_mvd3(circuit_path)
        worker.atlas = atlas
        worker.alpha = alpha
        worker.scales = scales
        worker.seed = seed
        worker.segment_type = segment_type
        choose_morphologies._fetch_atlas_data(  # pylint: disable=protected-access
            worker.atlas, worker.layer_names, memcache=True
        )
        cells_df = worker.cells.properties
        axon_morphs = pd.DataFrame()

        batch_size = int(
            len(cells_df.index) / (nb_jobs if nb_jobs > 0 else cpu_count())
        )
        L.info("Using batch_size of %s", batch_size)

        for gid, name in zip(
            cells_df.index,
            Parallel(
                nb_jobs,
                verbose=joblib_verbose,
                backend="multiprocessing",
                batch_size=batch_size,
            )(delayed(worker)(gid) for gid in cells_df.index),
        ):
            axon_morphs.loc[gid, "morphology"] = name
    else:
        L.info("Do not use placement algorithm for axons (use random choice instead)")

        cells_df = CellCollection.load_mvd3(circuit_path).properties
        axon_morphs = pd.DataFrame(index=cells_df.index, columns=["morphology"])

        morphs_df = pd.read_csv(morphs_df_path)
        if "use_axon" in morphs_df.columns:
            morphs_df = morphs_df.loc[morphs_df.use_axon]

        for mtype in tqdm(cells_df.mtype.unique()):
            all_cells = morphs_df[morphs_df.mtype == mtype]
            gids = cells_df[cells_df.mtype == mtype].index
            axon_morphs.loc[gids, "morphology"] = all_cells.sample(
                n=len(gids),
                replace=True,
                random_state=42,
            )["name"].to_list()

    axon_morphs.to_csv(axon_morphs_path, sep="\t", index=True)


def run_synthesize_morphologies(kwargs, nb_jobs=-1, debug_scales=None):
    """Run placement algorithm from python.

    Args:
        kwargs (dict): dictionary with argument from placement-algorithm CLI
    """
    parser_args = [
        i.replace("-", "_")
        for i in [
            "mvd3",
            "cells-path",
            "tmd-parameters",
            "tmd-distributions",
            "morph-axon",
            "base-morph-dir",
            "atlas",
            "atlas-cache",
            "seed",
            "out-mvd3",
            "out-apical",
            "out-morph-dir",
            "out-morph-ext",
            "max-files-per-dir",
            "overwrite",
            "max-drop-ratio",
            "no-mpi",
        ]
    ]

    # Setup defaults
    defaults = {
        "atlas_cache": None,
        "base_morph_dir": None,
        "max_drop_ratio": 0.0,
        "max_files_per_dir": None,
        "morph_axon": None,
        "mvd3": None,
        "out_morph_dir": "out",
        "out_morph_ext": ["swc"],
        "seed": 0,
    }

    # Set logging arguments
    logger_kwargs = None
    if debug_scales is not None:
        logger_kwargs = {
            "log_level": logging.DEBUG,
            "log_file": debug_scales.path,
        }

    # Run
    run_master(
        SynthesizeMorphologiesMaster,
        kwargs,
        parser_args,
        defaults,
        nb_jobs,
        logger_kwargs=logger_kwargs,
    )


def get_target_length(soma_layer, target_layer, cortical_thicknesses):
    """Compute the target length of a neurite from soma and target layer."""
    cortical_depths = np.insert(np.cumsum(cortical_thicknesses), 0, 0.0)
    soma_depth = np.mean(cortical_depths[soma_layer - 1 : soma_layer + 1])
    target_depth = cortical_depths[target_layer - 1]
    return soma_depth - target_depth


def get_max_len(neurite, mode="y"):
    """Get the max length of a neurite, either in y direction, or in radial direction."""
    max_len = 0
    for section in neurite.iter():
        if mode == "y":
            max_len = max(max_len, np.max(section.points[:, COLS.Y]))
        elif mode == "radial":
            max_len = max(max_len, np.max(np.linalg.norm(section.points, axis=1)))
        else:
            raise ValueError("mode must be in ['y', 'radial']")
    return max_len


def rescale_neurites(morphology, neurite_type, target_length, scaling_mode="y"):
    """Rescale neurites of morphologies to match target length."""
    max_length = -100
    for neurite in morphology.root_sections:
        if neurite.type == STR_TO_TYPES[neurite_type]:
            max_length = max(max_length, get_max_len(neurite, mode=scaling_mode))

    scale = target_length / max_length
    if 0.1 < scale < 10:
        for neurite in morphology.root_sections:
            if neurite.type == STR_TO_TYPES[neurite_type]:
                scale_section(
                    neurite,
                    ScaleParameters(),
                    ScaleParameters(mean=scale, std=0.0),
                    recursive=True,
                )

        return scale
    return None


def rescale_morphologies(
    morphs_df,
    scaling_rules,
    cortical_thicknesses,
    morphology_path="morphology_path",
    rescaled_morphology_base_path="rescaled_morphologies",
    rescaled_morphology_path="rescaled_morphology_path",
    ext=".h5",
    scaling_mode="y",
    skip_rescale=False,
):
    """Rescale all morphologies to fulfill scaling rules."""
    rescaled_morphology_base_path = Path(rescaled_morphology_base_path).absolute()
    if not rescaled_morphology_base_path.exists():
        os.mkdir(rescaled_morphology_base_path)

    for mtype in tqdm(morphs_df.mtype.unique()):
        gids = morphs_df[morphs_df.mtype == mtype].index

        if (
            not skip_rescale
            and mtype in scaling_rules
            and scaling_rules[mtype] is not None
        ):
            for neurite_type, target_layer in scaling_rules[mtype].items():
                soma_layer = int(mtype[1])
                target_layer = int(target_layer[1])
                target_length = get_target_length(
                    soma_layer=soma_layer,
                    target_layer=target_layer,
                    cortical_thicknesses=cortical_thicknesses,
                )
                for gid in gids:
                    morphology = Morphology(morphs_df.loc[gid, morphology_path])
                    scale = rescale_neurites(
                        morphology, neurite_type, target_length, scaling_mode
                    )
                    path = (
                        rescaled_morphology_base_path / morphs_df.loc[gid, "name"]
                    ).with_suffix(ext)
                    morphology.write(path)
                    morphs_df.loc[gid, rescaled_morphology_path] = path
                    morphs_df.loc[gid, neurite_type + "_scale"] = scale
        else:
            for gid in gids:
                morphology = Morphology(morphs_df.loc[gid, morphology_path])
                path = (
                    rescaled_morphology_base_path / morphs_df.loc[gid, "name"]
                ).with_suffix(ext)
                morphology.write(path)
                morphs_df.loc[gid, rescaled_morphology_path] = path
    return morphs_df


def _fit_population(mtype, file_names):
    # Load neurons
    if len(file_names) > 0:
        input_population = load_population(file_names)
    else:
        return (mtype, None, None)

    # Compute slope and intercept
    slope, intercept = fit_path_distance_to_extent(input_population)

    return mtype, slope, intercept


def add_scaling_rules_to_parameters(
    tmd_parameters,
    morphs_df_path,
    morphology_path,
    scaling_rules,
    nb_jobs=-1,
):
    """Add the scaling rules to TMD parameters"""

    def _get_target_layer(target_layer_str):
        res = re.match("^L?([0-9]*)", target_layer_str)
        if res is not None:
            return int(res.group(1))
        else:
            raise Exception("Scaling rule not understood: " + str(target_layer_str))

    def _process_scaling_rule(
        params, mtype, neurite_type, limits, default_limits, lim_type, default_fraction
    ):
        # Get the given limit or the default if not given
        if lim_type in limits:
            lim = limits[lim_type]
        elif lim_type in default_limits:
            lim = default_limits[lim_type]
        else:
            return

        # Update parameters
        layer = _get_target_layer(lim.get("layer"))
        fraction = lim.get("fraction", default_fraction)
        L.debug(
            "Add %s %s scaling rule to %s: %f in layer %i",
            neurite_type,
            lim_type,
            mtype,
            fraction,
            layer,
        )
        context = tmd_parameters[mtype].get("context_constraints", {})
        neurite_type_params = context.get(neurite_type, {})
        neurite_type_params.update({lim_type: {"layer": layer, "fraction": fraction}})
        context[neurite_type] = neurite_type_params
        params[mtype]["context_constraints"] = context

    # Add scaling rules to TMD parameters
    default_rules = scaling_rules.get("default") or {}

    for mtype in tmd_parameters.keys():
        mtype_rules = scaling_rules.get(mtype) or {}
        neurite_types = set(list(default_rules.keys()) + list(mtype_rules.keys()))
        for neurite_type in neurite_types:
            default_limits = default_rules.get(neurite_type) or {}
            limits = mtype_rules.get(neurite_type) or {}
            _process_scaling_rule(
                tmd_parameters,
                mtype,
                neurite_type,
                limits,
                default_limits,
                "hard_limit_min",
                0,
            )
            _process_scaling_rule(
                tmd_parameters,
                mtype,
                neurite_type,
                limits,
                default_limits,
                "extent_to_target",
                0.5,
            )
            _process_scaling_rule(
                tmd_parameters,
                mtype,
                neurite_type,
                limits,
                default_limits,
                "hard_limit_max",
                1,
            )

    # Build the file list for each mtype
    morphs_df = pd.read_csv(morphs_df_path)
    file_lists = [
        (mtype, morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list())
        for mtype in scaling_rules.keys()
        if mtype != "default"
    ]

    # Fit data and update TMD parameters
    for mtype, slope, intercept in Parallel(nb_jobs)(
        delayed(_fit_population)(mtype, file_names) for mtype, file_names in file_lists
    ):
        if slope is None or intercept is None:
            L.debug(
                "Fitting parameters not found for %s (slope=%s ; intercept=%s)",
                mtype,
                slope,
                intercept,
            )
            continue
        context = tmd_parameters[mtype].get("context_constraints", {})
        neurite_type_params = context.get("apical", {}).get("extent_to_target", {})
        neurite_type_params.update({"slope": slope, "intercept": intercept})
        context["apical"]["extent_to_target"] = neurite_type_params
        tmd_parameters[mtype]["context_constraints"] = context
        L.debug(
            "Fitting parameters for %s: slope=%s ; intercept=%s",
            mtype,
            slope,
            intercept,
        )
