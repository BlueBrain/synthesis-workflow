"""Functions for synthesis to be used by luigi tasks."""

import logging
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from morphio.mut import Morphology
from neuroc.scale import ScaleParameters
from neuroc.scale import scale_section
from neurom import load_morphology
from neurom.check.morphology_checks import has_apical_dendrite
from neurom.core.dataformat import COLS
from neurots import extract_input

try:
    from placement_algorithm.app.choose_morphologies import Master as ChooseMorphologyMaster

    with_placement_algo = True
except ImportError:
    with_placement_algo = False
from tmd.io.io import load_population
from tqdm import tqdm
from voxcell import CellCollection

from synthesis_workflow.fit_utils import fit_path_distance_to_extent
from synthesis_workflow.tools import run_master

L = logging.getLogger(__name__)
matplotlib.use("Agg")


def _get_morph_class(path):
    return "PC" if has_apical_dendrite(load_morphology(path)) else "IN"


def get_neurite_types(morphs_df):
    """Get the neurite types to consider for PC or IN cells by checking if apical exists."""
    morphs_df["morph_class"] = morphs_df["path"].apply(_get_morph_class)
    neurite_types = {}
    for mtype, _df in morphs_df.groupby("mtype"):
        morph_class = list(set(_df["morph_class"]))

        if len(morph_class) > 1:
            raise ValueError("f{mtype} has not consistent morph_class, we stop here")

        if morph_class[0] == "IN":
            neurite_types[mtype] = ["basal_dendrite"]
        if morph_class[0] == "PC":
            neurite_types[mtype] = ["basal_dendrite", "apical_dendrite"]
    return neurite_types


def apply_substitutions(original_morphs_df, substitution_rules=None):
    """Applies substitution rule on .dat file.

    Args:
        original_morphs_df (DataFrame): dataframe with morphologies
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies

    Returns:
        DataFrame: dataframe with original and new morphologies
    """
    if not substitution_rules:
        return original_morphs_df

    new_morphs_df = original_morphs_df.copy()
    for mtype_orig, mtypes in substitution_rules.items():
        for mtype in mtypes:
            df = original_morphs_df[original_morphs_df.mtype == mtype_orig].copy()
            df["mtype"] = mtype
            new_morphs_df = pd.concat([new_morphs_df, df])
    return new_morphs_df


def _build_distributions_single_mtype(
    mtype,
    morphs_df=None,
    neurite_types=None,
    diameter_model_function=None,
    config=None,
    morphology_path=None,
):
    """Internal function for multiprocessing of tmd_distribution building."""
    data = {}
    for neurite_type in neurite_types[mtype]:
        if "use_axon" in morphs_df.columns:
            morphology_paths = morphs_df.loc[
                (morphs_df.mtype == mtype) & morphs_df.use_axon, morphology_path
            ].to_list()
        else:
            morphology_paths = morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list()
        config["neurite_types"] = neurite_types[mtype]
        kwargs = {
            "neurite_types": neurite_types[mtype],
            "diameter_input_morph": morphology_paths,
        }
        if config["models"][0] != "simpler":
            config["diameter_model"] = partial(diameter_model_function, config=config)
        _data = extract_input.distributions(morphology_paths, **kwargs)

        data[neurite_type] = _data[neurite_type]
        data["diameter"] = _data["diameter"]
        data["soma"] = _data["soma"]
    return mtype, data


def build_distributions(
    mtypes,
    morphs_df,
    neurite_types,
    diameter_model_function,
    config,
    morphology_path,
    region,
    nb_jobs=-1,
    joblib_verbose=10,
):
    """Build tmd_distribution dictionary for synthesis.

    Args:
        mtypes (list): list of mtypes to build distribution for
        morphs_df (DataFrame): morphology dataframe with reconstruction to use
        diameter_model_function (function): diametrizer function (from diameter-synthesis)
        morphology_path (str): name of the column in morphs_df to use for paths to morphologies
        region (str): region we are building
        nb_jobs (int): number of jobs to run in parallal with joblib
        joblib_verbose (int): verbose level of joblib

    Returns:
        dict: dict to save to tmd_distribution.json
    """
    build_distributions_single_mtype = partial(
        _build_distributions_single_mtype,
        morphs_df=morphs_df,
        neurite_types=neurite_types,
        diameter_model_function=diameter_model_function,
        config=config,
        morphology_path=morphology_path,
    )
    tmd_distributions = {region: {}}
    for mtype, distribution in Parallel(nb_jobs, verbose=joblib_verbose)(
        delayed(build_distributions_single_mtype)(mtype) for mtype in mtypes
    ):
        tmd_distributions[region][mtype] = distribution
    return tmd_distributions


def get_axon_base_dir(morphs_df, col_name="morphology_path"):
    """Get the common base directory of morphologies."""
    if morphs_df.empty:
        raise RuntimeError("Can not get axon base dir from an empty DataFrame")

    col = morphs_df[col_name]

    # Get all parent directories
    parents = col.apply(lambda x: str(Path(x).parent)).unique()

    # Check they are all equal
    if len(parents) != 1:
        raise ValueError("Base dirs are different for axon grafting.")

    # Pick the first one
    axon_morphs_base_dir = parents[0]

    # Remove '-asc' suffix if present
    if axon_morphs_base_dir.split("-")[-1] == "asc":
        axon_morphs_base_dir = "-".join(axon_morphs_base_dir.split("-")[:-1])

    return axon_morphs_base_dir


def run_choose_morphologies(kwargs, nb_jobs=-1):
    """Runs placement algorithm from python.

    Args:
        kwargs (dict): dictionary with argument from placement-algorithm CLI
        nb_jobs (int): number of jobs
        debug_scales_path (str): path to the directory containing the log files
    """
    parser_args = [
        i.replace("-", "_")
        for i in [
            "mvd3",
            "cells-path",
            "morphdb",
            "atlas",
            "atlas-cache",
            "annotations",
            "rules",
            "segment-type",
            "alpha",
            "scales",
            "seed",
            "output",
            "no-mpi",
            "scores-output-path",
            "bias-kind",
            "no-optional-scores",
        ]
    ]

    # Setup defaults
    defaults = {
        "alpha": 1.0,
        "atlas_cache": None,
        "mvd3": None,
        "scales": None,
        "seed": 0,
        "segment-type": None,
        "scores-output-path": None,
        "bias-kind": "linear",
        "no-optional-scores": False,
    }

    # Set logging arguments
    logger_kwargs = None

    # Run
    run_master(
        ChooseMorphologyMaster,
        kwargs,
        parser_args,
        defaults,
        nb_jobs,
        logger_kwargs=logger_kwargs,
    )


# pylint: disable=too-many-arguments, too-many-locals
def create_axon_morphologies_tsv(
    circuit_path,
    morphs_df_path=None,
    atlas_path=None,
    annotations_path=None,
    rules_path=None,
    morphdb_path=None,
    alpha=1.0,
    scales=None,
    seed=0,
    axon_morphs_path="axon_morphs.tsv",
    scores_output_path=None,
    bias_kind="linear",
    with_optional_scores=True,
    nb_jobs=-1,
):
    """Create required axon_morphology tsv file for placement-algorithm to graft axons.

    Args:
        circuit_path (str): Path to circuit somata file
        morphs_df_path (str): Path to morphology dataframe
        atlas_path (str): Path to the atlas
        annotations_path (str): Path to annotations
        rules_path (str): Path to rules
        morphdb_path (str): Path to morphdb file
        alpha (float): Use `score ** alpha` as morphology choice probability
        scales (list(float)): Scale(s) to check
        seed (int): Random number generator seed
        axon_morphs_path (str): Name of the axon morphology list in .tsv format
        scores_output_path (str): Make ``placement_algorithm.app.choose_morphologies`` export scores
            into files in this folder
        bias_kind (str): Kind of bias used to penalize scores of rescaled morphologies
            (can be "linear" or "gaussian")
        with_optional_scores (bool): Use or ignore optional rules for morphology choice
        nb_jobs (int): Number of jobs
    """
    check_placement_params = {
        "morphs_df_path": morphs_df_path is None,
        "atlas_path": atlas_path is not None,
        "annotations_path": annotations_path is not None,
        "rules_path": rules_path is not None,
        "morphdb_path": morphdb_path is not None,
    }
    if any(check_placement_params.values()) and not all(check_placement_params.values()):
        _params = [k for k in check_placement_params if k != "morphs_df_path"]
        raise ValueError(
            f"Either 'morphs_df_path' or all the following parameter should be None: {_params}"
        )

    if all(check_placement_params.values()) and with_placement_algo:
        L.info("Use placement algorithm for axons")

        kwargs = {
            "cells-path": circuit_path,
            "morphdb": morphdb_path,
            "atlas": atlas_path,
            "annotations": annotations_path,
            "rules": rules_path,
            "segment-type": "axon",
            "alpha": alpha,
            "scales": scales,
            "seed": seed,
            "output": axon_morphs_path,
            "scores-output-path": scores_output_path,
            "bias-kind": bias_kind,
            "no-optional-scores": not bool(with_optional_scores),
            "no-mpi": True,
        }

        run_choose_morphologies(kwargs, nb_jobs=nb_jobs)

    else:
        L.info("Do not use placement algorithm for axons (use random choice instead)")

        cells_df = CellCollection.load(circuit_path).properties
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

        axon_morphs.to_csv(axon_morphs_path, sep="\t", na_rep="N/A")


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
        if neurite.type.name == neurite_type:
            max_length = max(max_length, get_max_len(neurite, mode=scaling_mode))

    scale = target_length / max_length
    if 0.1 < scale < 10:
        for neurite in morphology.root_sections:
            if neurite.type.name == neurite_type:
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

        if not skip_rescale and mtype in scaling_rules and scaling_rules[mtype] is not None:
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
                    scale = rescale_neurites(morphology, neurite_type, target_length, scaling_mode)
                    path = (rescaled_morphology_base_path / morphs_df.loc[gid, "name"]).with_suffix(
                        ext
                    )
                    morphology.write(path)
                    morphs_df.loc[gid, rescaled_morphology_path] = path
                    morphs_df.loc[gid, neurite_type + "_scale"] = scale
        else:
            for gid in gids:
                morphology = Morphology(morphs_df.loc[gid, morphology_path])
                path = (rescaled_morphology_base_path / morphs_df.loc[gid, "name"]).with_suffix(ext)
                morphology.write(path)
                morphs_df.loc[gid, rescaled_morphology_path] = path
    return morphs_df


def _fit_population(mtype, neurite_type, file_names):
    # Load neurons
    if len(file_names) > 0:
        input_population = load_population(file_names, use_morphio=True)
    else:
        return (mtype, neurite_type, None, None)

    # Compute slope and intercept
    try:
        slope, intercept = fit_path_distance_to_extent(input_population, neurite_type=neurite_type)
    except IndexError:
        return (mtype, neurite_type, None, None)

    return mtype, neurite_type, slope, intercept


def add_scaling_rules_to_parameters(
    tmd_parameters,
    morphs_df_path,
    morphology_path,
    scaling_rules,
    nb_jobs=-1,
):
    """Adds the scaling rules to TMD parameters."""

    def _get_target_layer(target_layer_str):
        res = re.match("^L?([0-9]*)", target_layer_str)
        if res is not None:
            return int(res.group(1))
        else:
            raise ValueError("Scaling rule not understood: " + str(target_layer_str))

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
    neurite_types_map = defaultdict(list)
    for mtype in tmd_parameters.keys():
        mtype_rules = scaling_rules.get(mtype) or {}
        neurite_types = set(list(default_rules.keys()) + list(mtype_rules.keys()))
        for neurite_type in neurite_types:
            default_limits = default_rules.get(neurite_type) or {}
            limits = mtype_rules.get(neurite_type) or {}

            if "extent_to_target" in limits:
                neurite_types_map[mtype].append(neurite_type)

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
        (mtype, neurite_type, morphs_df.loc[morphs_df.mtype == mtype, morphology_path].to_list())
        for mtype in scaling_rules.keys()
        for neurite_type in neurite_types_map[mtype]
        if mtype != "default"
    ]

    L.debug("Number of files: %s", [(t, len(f)) for t, _, f in file_lists])

    # Fit data and update TMD parameters
    for mtype, neurite_type, slope, intercept in Parallel(nb_jobs)(
        delayed(_fit_population)(mtype, neurite_type, file_names)
        for mtype, neurite_type, file_names in file_lists
    ):
        if slope is None or intercept is None:
            L.debug(
                "Fitting parameters not found for neurite_type %s and %s (slope=%s ; intercept=%s)",
                neurite_type,
                mtype,
                slope,
                intercept,
            )
            continue
        context = tmd_parameters[mtype].get("context_constraints", {})
        neurite_type_params = context.get(neurite_type, {}).get("extent_to_target", {})
        neurite_type_params.update({"slope": slope, "intercept": intercept})
        context[neurite_type]["extent_to_target"] = neurite_type_params
        tmd_parameters[mtype]["context_constraints"] = context
        L.debug(
            "Fitting parameters for neurite_type %s and %s: slope=%s ; intercept=%s",
            neurite_type,
            mtype,
            slope,
            intercept,
        )
