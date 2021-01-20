"""Functions to create and mofify combos dataframe."""
from pathlib import Path
import json
import logging
import pandas as pd

from bluepymm.prepare_combos.parse_files import read_mm_recipe

L = logging.getLogger(__name__)


def get_me_types_map(recipe, emodel_etype_map):
    """Use recipe data and bluepymm to get mtype/etype combos."""
    me_types_map = pd.DataFrame()
    for i in recipe.index:
        combo = recipe.loc[i]
        for emodel, emap in emodel_etype_map.items():
            if combo.layer in emap["layer"] and combo.etype == emap["etype"]:
                if "mtype" in emap:
                    if emap["mtype"] == combo.fullmtype:
                        combo["emodel"] = emodel
                        me_types_map = me_types_map.append(combo.copy())
                else:
                    combo["emodel"] = emodel
                    me_types_map = me_types_map.append(combo.copy())

    return me_types_map.rename(columns={"fullmtype": "mtype"}).reset_index()


def create_morphs_combos_df(
    morphs_df,
    recipe_path=None,
    emodel_etype_map_path=None,
    emodels=None,
    me_types_map=None,
):
    """From the morphs_df, create a dataframe with all possible combos."""
    if me_types_map is None and emodel_etype_map_path is not None and recipe_path is not None:
        recipe = read_mm_recipe(recipe_path)
        emodel_etype_map = json.load(open(emodel_etype_map_path, "rb"))
        me_types_map = get_me_types_map(recipe, emodel_etype_map)

    morphs_combos_df = pd.DataFrame()
    for combo_id in me_types_map.index:
        if emodels is not None:
            if me_types_map.loc[combo_id, "emodel"] not in emodels:
                continue
        combo = morphs_df[morphs_df.mtype == me_types_map.loc[combo_id, "mtype"]]
        combo = combo.assign(etype=me_types_map.loc[combo_id, "etype"])
        combo = combo.assign(emodel=me_types_map.loc[combo_id, "emodel"])
        morphs_combos_df = morphs_combos_df.append(combo.copy())

    morphs_combos_df = (
        morphs_combos_df.drop_duplicates().reset_index().rename(columns={"index": "morph_gid"})
    )
    return morphs_combos_df


def _base_emodel(emodel):
    return "_".join(emodel.split("_")[:2])


def add_for_optimisation_flag(config_path, morphs_combos_df=None, morphs_df=None, emodels=None):
    """Add for_optimisation flag for combos used for optimisation."""
    if morphs_df is None and morphs_combos_df is None:
        raise Exception("Please provide at least one dataframe.")

    if morphs_combos_df is not None:
        emodels = list(set(morphs_combos_df.emodel))
        morphs_combos_df["for_optimisation"] = False
        for emodel in emodels:
            recipe = json.load(
                open(
                    config_path / _base_emodel(emodel) / "config/recipes/recipes.json",
                    "rb",
                )
            )[_base_emodel(emodel)]
            opt_mask = (morphs_combos_df.emodel == emodel) & (
                morphs_combos_df.name == Path(recipe["morphology"][0][1]).stem
            )
            morphs_combos_df.loc[opt_mask, "for_optimisation"] = True
            if len(morphs_combos_df[opt_mask]) == 0:

                new_combo = morphs_combos_df[
                    (morphs_combos_df.name == Path(recipe["morphology"][0][1]).stem)
                    & (morphs_combos_df.for_optimisation == 1)
                ]
                if len(new_combo) > 0:
                    new_combo = new_combo.iloc[0]
                    L.warning("Duplicate optimisation cell for emodel %s", emodel)
                else:
                    L.warning("Error, no cell for %s", emodel)

                new_combo["emodel"] = emodel
                new_combo["etype"] = emodel.split("_")[0]
                morphs_combos_df = morphs_combos_df.append(new_combo.copy())

    if morphs_df is not None:
        morphs_df["for_optimisation"] = False
        if emodels is None and morphs_combos_df is None:
            raise Exception("Please provide a list of emodels for your cells")
        for emodel in emodels:
            recipe = json.load(
                open(
                    config_path / _base_emodel(emodel) / "config/recipes/recipes.json",
                    "rb",
                )
            )[_base_emodel(emodel)]
            morphs_df.loc[
                (morphs_df.name == Path(recipe["morphology"][0][1]).stem),
                "for_optimisation",
            ] = True
    return morphs_combos_df, morphs_df


def add_for_optimisation_flag_old(config_path, morphs_combos_df=None, morphs_df=None, emodels=None):
    """Add for_optimisation flag for combos used for optimisation."""
    if morphs_df is None and morphs_combos_df is None:
        raise Exception("Please provide at least one dataframe.")

    if morphs_combos_df is not None:
        emodels = list(set(morphs_combos_df.emodel))
        morphs_combos_df["for_optimisation"] = False
        for emodel in emodels:
            recipe = json.load(open(config_path / emodel / "recipes/recipes.json", "rb"))[emodel]
            morphs_combos_df.loc[
                (morphs_combos_df.emodel == emodel)
                & (morphs_combos_df.name == Path(recipe["morphology"][0][1]).stem),
                "for_optimisation",
            ] = True
            if (
                len(
                    morphs_combos_df.loc[
                        (morphs_combos_df.emodel == emodel)
                        & (morphs_combos_df.name == Path(recipe["morphology"][0][1]).stem)
                    ]
                )
                == 0
            ):
                L.warning("Could not find a cell for optimisation for emodel %s", emodel)

    if morphs_df is not None:
        morphs_df["for_optimisation"] = False
        if emodels is None and morphs_combos_df is None:
            raise Exception("Please provide a list of emodels for your cells")
        for emodel in emodels:
            recipe = json.load(open(config_path / emodel / "recipes/recipes.json", "rb"))[emodel]
            morphs_df.loc[
                (morphs_df.name == Path(recipe["morphology"][0][1]).stem),
                "for_optimisation",
            ] = True
