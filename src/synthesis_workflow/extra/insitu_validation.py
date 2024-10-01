"""In-situ validation functions for synthesis in atlas."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
import seaborn as sns
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from matplotlib.backends.backend_pdf import PdfPages
from morphio import Morphology
from morphio import SectionType
from region_grower.atlas_helper import AtlasHelper
from tqdm import tqdm
from voxcell import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from synthesis_workflow.circuit import get_layer_tags

matplotlib.use("Agg")


def _extent_function(row):
    """Inner function to compute the max extent of dendrites."""
    neuron = nm.load_morphology(row["path"])
    a = nm.get("max_radial_distance", neuron, neurite_type=nm.NeuriteType.basal_dendrite)
    b = nm.get("max_radial_distance", neuron, neurite_type=nm.NeuriteType.apical_dendrite)
    return {"extent": max(a, b)}


def compute_extents(sonata_path, morphology_path, ext=".h5", parallel_factory="multiprocessing"):
    """Compute the extents (max radial distance) of all morphologies.

    Args:
        sonata_path (str): path to a sonata file with morphologies
        morphology_path (str): base path to the morphologies
        ext (str): extension of the morphologies
        parallel_factory (str): name of parallel factory to use (see BluePyParallel)

    Returns:
        pandas.DataFrame: dataframe from sonata with column 'extent' containing the computed values
    """
    df = CellCollection.load_sonata(sonata_path).as_dataframe()

    df["path"] = morphology_path + df["morphology"] + ext
    _df = df[["path", "mtype", "etype"]]

    factory = init_parallel_factory(parallel_factory)
    df["extent"] = evaluate(
        _df, _extent_function, new_columns=[["extent", 0]], parallel_factory=factory
    )["extent"]
    return df


def plot_extents(results_df, pdf_name="extents.pdf", bins=200):
    """Plot histograms of extents of morphologies.

    Args:
        results_df (pandas.Dataframe): results from compute_extents
        pdf_name (str): name of pdf to save
        bins (int): number of bins for histograms
    """
    with PdfPages(pdf_name) as pdf:
        for mtype, _df in tqdm(results_df.groupby("mtype")):
            plt.figure()
            plt.hist(_df["extent"].to_list(), bins=bins)
            plt.suptitle(f"mtype: {mtype}")
            plt.xlabel("extent (microns)")
            plt.ylabel("number of morphologies")
            pdf.savefig()


# pylint: disable=too-many-locals
def get_layer_morpho_counts(
    sonata_path,
    morphology_path,
    atlas_path,
    n_cells=100,
    section_type="basal_dendrite",
    region=None,
    hemisphere="right",
    ext=".h5",
    region_structure_path="region_structure.yaml",
):
    """For each layer, compute the fraction of crossing morphologies.

    Args:
        sonata_path (str): path to sonata file
        morphology_path (str): base path to morphologies
        atlas_path (str): path to atlas folder
        n_cells (int): number of morphologies to use to compute the fraction
        section_type (str): section type (attr of morphio.SectionType)
        region (str): is not None, must be the name of a region to filter morphologies
        hemisphere (str): left/right hemisphere
        ext (str): extension of morphology files
        region_structure_path (str): path to region_structure.yaml file

    Returns:
        pandas.DataFrame: dafaframe with the counts, mtype and layers information
    """
    section_type = getattr(SectionType, section_type)
    cells = CellCollection.load(sonata_path).as_dataframe()
    morph_path = Path(morphology_path)

    layer_annotation, _ = get_layer_tags(atlas_path, region_structure_path)

    dfs = []
    for mtype in tqdm(cells.mtype.unique()):
        _cells = cells[cells.mtype == mtype]
        if region is not None:
            _cells = _cells[_cells.region == f"{region}@{hemisphere}"]
        _n_cells = min(n_cells, len(_cells.index))
        _cells = _cells.sample(_n_cells)

        _layer_count = np.zeros(7)
        for gid in _cells.index:
            morph = Morphology((morph_path / _cells.loc[gid, "morphology"]).with_suffix(ext))
            points = []
            for section in morph.iter():
                if section.type == section_type:
                    points += (
                        section.points + _cells.loc[gid, ["x", "y", "z"]].to_numpy()[np.newaxis]
                    ).tolist()

            if len(points) > 0:
                layers = layer_annotation.lookup(points, outer_value=0)
                _layers, _counts = np.unique(layers, return_counts=True)
                _layer_count[_layers] += 1
        _df = pd.DataFrame()
        _df["count"] = _layer_count / _n_cells
        _df["layer"] = _df.index
        _df["mtype"] = mtype
        dfs.append(_df)
    return pd.concat(dfs)


def plot_layer_morph_counts(counts_df, pdf_name="layer_count_comparison.pdf"):
    """Plot count of crossing of morphologies with layers.

    If 'label' column is present in counts_df, the barplots will split the data for comparison.

    For example:
        df1 = get_layer_morpho_counts(...)
        df1['label'] = 'dataset1'
        df2 = get_layer_morpho_counts(...)
        df2['label'] = 'dataset2'
        plot_layer_morph_counts(pd.concat([df1, df2]))

    Args:
        counts_df (pandas.DataFrame): results of get_layer_morpho_counts,
        pdf_name (str): name of pdf to save
    """
    with PdfPages(pdf_name) as pdf:
        for mtype in counts_df.mtype.unique():
            df = counts_df[counts_df.mtype == mtype]
            if len(df[df["count"] > 0]) > 0:
                plt.figure(figsize=(5, 4))
                if "label" in df.columns:
                    sns.barplot(x="layer", hue="label", y="count", data=df)
                else:
                    sns.barplot(x="layer", y="count", data=df)
                plt.axhline(1.0)
                plt.suptitle(mtype)
                pdf.savefig()
                plt.close()


def get_depth(atlas, pos, layers, lower=0, upper=1000, precision=0.1):
    """Estimate the depth from 'pos' to pia using layer annotation from atlas."""
    orientation = atlas.orientations.lookup(pos)[0].dot([0, 1, 0])
    voxel_dim = precision * layers.voxel_dimensions.max()

    def _get_layer(_pos):
        return layers.lookup(pos + _pos * orientation, outer_value=0)

    def _search(lower, upper, search_depth):
        mid = 0.5 * (lower + upper)
        layer = _get_layer(mid)

        if upper - lower < voxel_dim:
            return np.linalg.norm(mid * orientation)

        if layer >= 1:
            return _search(mid, upper, search_depth + 1)
        if layer == 0:
            return _search(lower, mid, search_depth + 1)
        return None

    upper_layer = _get_layer(upper)
    if upper_layer > 0:
        return get_depth(atlas, pos, layers, lower=lower, upper=2 * upper)
    return _search(lower, upper, 0)


# pylint: disable=too-many-arguments,too-many-locals
def plot_layer_collage(
    sonata_path,
    morphology_path,
    atlas_path,
    n_cells=10,
    section_types=None,
    region=None,
    hemisphere="right",
    pdf_name="layer_collage.pdf",
    shift=200,
    dpi=200,
    ext=".h5",
    region_structure_path="region_structure.yaml",
):
    """Plot morphologies next to each other with points colored by layer.

    Args:
        sonata_path (str): path to sonata file
        morphology_path (str): base path to morphologies
        atlas_path (str): path to atlas folder
        n_cells (int): number of morphologies to use to compute the fraction
        section_types (list): list of section types (attr of morphio.SectionType),
            if None, basal_dendrite, apical_dendrite will be used
        region (str): is not None, must be the name of a region to filter morphologies
        hemisphere (str): left/right hemisphere
        pdf_name (str): name of pdf to save
        shift (float): x-shift between morphologies
        dpi (int): dpi to save rasterized pdf
        ext (str): extension of morphology files
        region_structure_path (str): path to region_structure.yaml file
    """
    if section_types is None:
        section_types = ["basal_dendrite", "apical_dendrite"]
    section_types = [getattr(SectionType, section_type) for section_type in section_types]

    cells = CellCollection.load(sonata_path).as_dataframe()
    morph_path = Path(morphology_path)
    layer_annotation, _ = get_layer_tags(atlas_path, region_structure_path)
    atlas = AtlasHelper(Atlas().open(atlas_path), region_structure_path=region_structure_path)
    region_depths = atlas.compute_region_depth(region)

    cmap = matplotlib.colors.ListedColormap(["C0", "C1", "C2", "C3", "C4", "C5", "C6"])
    with PdfPages(pdf_name) as pdf:
        for mtype in tqdm(cells.mtype.unique()):
            _cells = cells[cells.mtype == mtype]
            if region is not None:
                _cells = _cells[_cells.region == f"{region}@{hemisphere}"]
            _n_cells = min(n_cells, len(_cells.index))
            _cells = _cells.sample(_n_cells)

            plt.figure(figsize=(len(_cells.index) * 2, 3))
            for i, gid in enumerate(_cells.index):
                morph = Morphology((morph_path / _cells.loc[gid, "morphology"]).with_suffix(ext))
                points = []
                for section in morph.iter():
                    if section.type in section_types:
                        points += section.points.tolist()

                pos = _cells.loc[gid, ["x", "y", "z"]].to_numpy()
                points = np.array(points)
                _points = points + pos[np.newaxis]
                layers = layer_annotation.lookup(_points, outer_value=0)
                depth = region_depths.lookup(pos)

                rotation = atlas.orientations.lookup(pos)[0]

                points = points.dot(rotation.T)
                points += np.array([shift * i, 0, 0])[np.newaxis]
                scat = plt.scatter(
                    *points[:, :2].T, c=layers, s=0.1, cmap=cmap, vmin=-0.5, vmax=6.5
                )
                plt.scatter(shift * i, 0, c="k", s=5)
                plt.plot([shift * i - shift / 2, shift * i + shift / 2], [depth, depth], "-k")
                new_depth = get_depth(atlas, pos, layer_annotation)
                plt.plot(
                    [shift * i - shift / 2, shift * i + shift / 2], [new_depth, new_depth], "-r"
                )

            plt.colorbar(scat, label="layers (0=outside)")
            plt.suptitle(mtype)
            plt.axis("equal")
            plt.gca().set_rasterized(True)
            pdf.savefig(bbox_inches="tight", dpi=dpi)
            plt.close()
