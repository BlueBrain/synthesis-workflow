"""Functions for synthesis to be used by luigi tasks."""
import json
import logging
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from joblib import delayed
from joblib import Parallel
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
from tns import extract_input
from voxcell import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from . import STR_TO_TYPES
from .tools import run_master


L = logging.getLogger("luigi-interface")
matplotlib.use("Agg")


def get_neurite_types(pc_in_types_path, mtypes):
    """Get the neurite types to consider for PC or IN cells."""
    with open(pc_in_types_path, "rb") as pc_in_file:
        pc_in_files = yaml.full_load(pc_in_file)

    return {
        mtype: ["basal"] if pc_in_files[mtype] == "IN" else ["basal", "apical"]
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
        "metadata": {"cortical_thickness": json.loads(cortical_thickness)},
    }
    for mtype, distribution in Parallel(nb_jobs)(
        delayed(build_distributions_single_mtype)(mtype) for mtype in tqdm(mtypes)
    ):
        tmd_distributions["mtypes"][mtype] = distribution
    return tmd_distributions


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
):
    """Create required axon_morphology tsv file for placement-algorithm to graft axons.

    Args:
        circuit_path (str): path to circuit somata file
        morphs_df_path (str): path to morphology dataframe
        axon_morphs_path (str): name of the axon morphology list in .tsv format

    Returns:
        str: path to base directory of morphologies for axon grafting
    """
    morphs_df = pd.read_csv(morphs_df_path)

    check_placement_params = [
        atlas_path is not None,
        annotations_path is not None,
        rules_path is not None,
        morphdb_path is not None,
    ]
    if any(check_placement_params) and not all(check_placement_params):
        raise ValueError(
            "Either 0 or all the following parameter should be None: %s"
            % [
                "atlas_path",
                "annotations_path",
                "rules_path",
                "morphdb_path",
            ]
        )

    if all(check_placement_params):
        use_placement = True
        L.info("Use placement algorithm for axons")
    else:
        use_placement = False
        L.info("Do not use placement algorithm for axons (use random choice instead)")

    if use_placement:
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
        cells_df = worker.cells.as_dataframe()
    else:
        cells_df = CellCollection.load_mvd3(circuit_path).as_dataframe()

    axon_morphs_base_dir = None
    axon_morphs = pd.DataFrame()
    for gid in cells_df.index:
        all_cells = morphs_df[
            (morphs_df.mtype == cells_df.loc[gid, "mtype"]) & morphs_df.use_axon
        ]
        if len(all_cells) > 0:
            if use_placement:
                cell = all_cells.loc[all_cells["name"] == worker(gid)].iloc[0]
            else:
                cell = all_cells.sample().iloc[0]

            axon_morphs.loc[gid, "morphology"] = cell["name"]

            if axon_morphs_base_dir is None:
                axon_morphs_base_dir = str(Path(cell["morphology_path"]).parent)
            elif axon_morphs_base_dir != str(Path(cell["morphology_path"]).parent):
                raise Exception("Base dirs are different for axon grafting.")
        else:
            L.info("Axon grafting: no cells for %s", cells_df.loc[gid, "mtype"])

    axon_morphs.index -= 1
    axon_morphs.to_csv(axon_morphs_path, sep="\t", index=True)
    if (
        axon_morphs_base_dir is not None
        and axon_morphs_base_dir.split("-")[-1] == "asc"
    ):
        axon_morphs_base_dir = "-".join(axon_morphs_base_dir.split("-")[:-1])
    return axon_morphs_base_dir


def run_synthesize_morphologies(kwargs, nb_jobs=-1):
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

    # Run
    run_master(SynthesizeMorphologiesMaster, kwargs, parser_args, defaults, nb_jobs)


def get_mean_neurite_lengths(
    morphs_df,
    neurite_type="apical",
    mtypes=None,
    morphology_path="morphology_path",
    percentile=None,
):
    """Extract the mean radial neurite lengths of a population, by mtypes."""
    if mtypes is None:
        mtypes = ["all"]
    if mtypes[0] != "all":
        morphs_df = morphs_df[morphs_df.mtype.isin(mtypes)]

    # Choose mean or percentile function
    def _percentile(q, a, *args, **kwargs):
        return np.percentile(a, q, *args, **kwargs)

    if percentile is None:
        f = np.mean
    else:
        f = partial(_percentile, float(percentile))

    apical_lengths = defaultdict(list)
    for gid in tqdm(morphs_df.index):
        neuron = Morphology(morphs_df.loc[gid, morphology_path])
        for neurite in neuron.root_sections:
            if neurite.type == STR_TO_TYPES[neurite_type]:
                apical_lengths[morphs_df.loc[gid, "mtype"]].append(get_max_len(neurite))

    return {mtype: float(f(lengths)) for mtype, lengths in apical_lengths.items()}


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

        if mtype in scaling_rules and scaling_rules[mtype] is not None:
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
                    if not skip_rescale:
                        scale = rescale_neurites(
                            morphology, neurite_type, target_length, scaling_mode
                        )
                    else:
                        scale = None
                    if scale is None:
                        L.debug(
                            "The morphology '%s', with mtype '%s' was not rescaled",
                            morphs_df.loc[gid, "name"],
                            mtype,
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


def add_scaling_rules_to_parameters(
    tmd_parameters, mean_lengths, scaling_rules, hard_limits
):
    """Add the scaling rules to TMD parameters"""

    def _get_target_layer(target_layer_str):
        if re.match("^L", target_layer_str):
            position = 0.5
            if len(target_layer_str) == 3:
                position = 1
            return int(target_layer_str[1]), position
        raise Exception("Scaling rule not understood: " + str(target_layer_str))

    for neurite_type in mean_lengths:
        for mtype, mean_length in mean_lengths[neurite_type].items():
            if (
                mtype in scaling_rules
                and scaling_rules[mtype] is not None
                and neurite_type in scaling_rules[mtype]
            ):
                tmd_parameters[mtype][neurite_type]["expected_max_length"] = mean_length
                layer, position = _get_target_layer(scaling_rules[mtype][neurite_type])
                tmd_parameters[mtype][neurite_type]["target_layer"] = layer
                tmd_parameters[mtype][neurite_type]["target_layer_position"] = position

    for neurite_type, limits in hard_limits.items():
        if limits is None:
            continue
        if "hard_limits" not in tmd_parameters[neurite_type]:
            tmd_parameters[neurite_type]["hard_limits"] = defaultdict(dict)
        if "min" in limits:
            lim = limits["min"]
            min_layer = int(lim.get("layer")[1:])
            min_fraction = lim.get("fraction", 0.0)
            tmd_parameters[neurite_type]["hard_limits"]["min"]["layer"] = min_layer
            tmd_parameters[neurite_type]["hard_limits"]["min"][
                "fraction"
            ] = min_fraction
            L.debug(
                "Add min hard limit to %s: %i in %f layer",
                neurite_type,
                min_layer,
                min_fraction,
            )
        if "max" in limits:
            lim = limits["max"]
            max_layer = int(lim.get("layer")[1:])
            max_fraction = lim.get("fraction", 1.0)
            tmd_parameters[neurite_type]["hard_limits"]["max"]["layer"] = max_layer
            tmd_parameters[neurite_type]["hard_limits"]["max"][
                "fraction"
            ] = max_fraction
            L.debug(
                "Add max hard limit to %s: %i in %f layer",
                neurite_type,
                max_layer,
                max_fraction,
            )
