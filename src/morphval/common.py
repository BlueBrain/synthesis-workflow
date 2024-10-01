"""Private helper functions of the validation module."""

import contextlib
import json
import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from neurom import NeuriteType
from neurom import geom
from neurom.core.dataformat import COLS
from neurom.core.morphology import iter_neurites
from neurom.core.population import Population
from neurom.core.types import tree_type_checker as is_type
from neurom.view import matplotlib_impl
from neurom.view import matplotlib_utils
from region_grower.utils import NumpyEncoder

COMP_MAP = {
    "basal_dendrite": NeuriteType.basal_dendrite,
    "apical_dendrite": NeuriteType.apical_dendrite,
    "axon": NeuriteType.axon,
    "soma": None,
}


def pretty_name(name):
    """Make a pretty name."""
    return name.replace("_", " ")


def add_progress_bar(items, template, description):
    """Add a progress bar in notebooks."""
    if description:
        from tqdm import tqdm_notebook  # pylint: disable=import-outside-toplevel

        return tqdm_notebook(items, desc=template.format(description))
    return items


def progress_bar_label(template, description, notebook):
    """Format progress bar label."""
    return template.format(description) if notebook else None


def dump2json(data_dir, var_name, data):
    """Save the dictionary 'data' into a .json file.

    Args:
        data_dir : the data directory
        var_name : the name of the dictionary as a string
        data : the data dictionary
    """
    fname = os.path.join(data_dir, var_name + ".json")
    with open(fname, "w", encoding="utf-8") as fd:
        json.dump(data, fd, cls=NumpyEncoder, indent=2, sort_keys=True)
    return fname


def load_json(fname):
    """Load a json file from file with fname into the results dictionary.

    Returns:
        The results dictionary
    """
    with open(fname, "r", encoding="utf-8") as fd:
        return json.load(fd)


def find_cells(dir_name):
    """Return the cells in dir_name.

    Args:
        dir_name (str): path to the directory containing the cells

    Returns:
        The list of cell files
    """
    # now we skip all hidden files, starting with '.'
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f[0] != "."]

    if not files:
        raise ValueError("There are no cells in '" + dir_name + "'")

    return files


@contextlib.contextmanager
def pyplot_non_interactive():
    """Suppress pyplot showing blank graphs when in interactive mode.

    This usually happens in the context of jupyter notebooks
    """
    if plt.isinteractive():
        plt.ioff()
        yield
        plt.ion()
    else:
        yield


@contextlib.contextmanager
def get_agg_fig():
    """Yield a figure and close it afterwards."""
    fig = Figure()
    FigureCanvas(fig)
    yield fig
    plt.close(fig)


def center_population(population):
    """Return a new population where all cells have been translated to their soma origins."""
    morphs = [geom.transform.translate(n, -1 * np.asarray(n.soma.center)) for n in population]
    return Population(morphs, name=population.name)


def truncate_population(population, count):
    """Return a new population with `count` elements."""
    morphs = [population[i] for i in range(min(len(population), count))]
    return Population(morphs, name=population.name)


def get_components_population(population, component):
    """Get component of a given population."""
    if component == "full_morph":
        return population

    def filtered_neurites(n):
        return iter_neurites(n, filt=is_type(COMP_MAP[component]))

    morphs = []
    for n in population:
        nrn = deepcopy(n)
        for neurite in filtered_neurites(n):
            if neurite not in nrn.neurites:
                nrn.delete_section(neurite, recursive=True)

        nrn.name = n.name + "_" + component
        morphs.append(nrn)

    return Population(morphs, name=population.name)


def compute_bounding_box(*populations):
    """Get bounding box of a population."""
    min_bounding_box = np.full(shape=(3,), fill_value=np.inf)
    max_bounding_box = np.full(shape=(3,), fill_value=-np.inf)

    for population in populations:
        for morph in population:
            bounding_box = geom.bounding_box(morph)
            min_bounding_box = np.minimum(min_bounding_box, bounding_box[0][COLS.XYZ])
            max_bounding_box = np.maximum(max_bounding_box, bounding_box[1][COLS.XYZ])

    return (
        (min_bounding_box[COLS.X], max_bounding_box[COLS.X]),
        (min_bounding_box[COLS.Y], max_bounding_box[COLS.Y]),
    )


def plot_population(output_dir, population, xlim, ylim, notebook_desc=None):
    """Plot a population."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ret = []
    with pyplot_non_interactive():
        for morph in add_progress_bar(population, "-- {}", notebook_desc):
            try:
                fig, ax = matplotlib_utils.get_figure()
                matplotlib_impl.plot_morph(morph, ax, diameter_scale=None)
                matplotlib_utils.plot_style(fig=fig, ax=ax)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                file_name = os.path.join(output_dir, morph.name + ".png")
                fig.savefig(file_name)
                plt.close(fig)
                ret.append(file_name)
            except Exception:  # pylint: disable=broad-except
                print("Failed to plot neuron: ", morph.name)
    return ret


def plot_normalized_neurons(
    output_dir,
    ref_population,
    test_population,
    cell_figure_count,
    components,
    notebook_desc=None,
):
    """Plot cell examples and store the file in output_dir.

    Args:
        output_dir(str): path to the output directory
        ref_population(NeuroM morph population): reference population
        test_population(NeuroM morph population): test population
        cell_figure_count(int): number of exemplars to plot
        components(list of str): components to visualize
        notebook_desc(str): string used for labelling notebook progress bar
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_population = _center_truncate_population(ref_population, cell_figure_count)
    test_population = _center_truncate_population(test_population, cell_figure_count)

    ref_output_dir = os.path.join(output_dir, "ref")
    test_output_dir = os.path.join(output_dir, "test")

    ref_plot_paths = {}
    test_plot_paths = {}

    comp_dict = {"full_morph": "morphologies"}
    for comp in components:
        comp_dict.update({comp: pretty_name(comp) + "s"})

    for comp in add_progress_bar(comp_dict, "[{}] Plot morphologies", notebook_desc):
        ref_comp_pop = get_components_population(ref_population, comp)
        test_comp_pop = get_components_population(test_population, comp)
        xlim, ylim = compute_bounding_box(ref_comp_pop, test_comp_pop)

        desc = progress_bar_label("Reference {}", pretty_name(comp_dict[comp]), notebook_desc)
        ref_plot_paths[comp] = plot_population(ref_output_dir, ref_comp_pop, xlim, ylim, desc)
        desc = progress_bar_label("Test {}", pretty_name(comp_dict[comp]), notebook_desc)
        test_plot_paths[comp] = plot_population(test_output_dir, test_comp_pop, xlim, ylim, desc)

    return ref_plot_paths, test_plot_paths


def _center_truncate_population(population, cell_figure_count):
    return truncate_population(center_population(population), cell_figure_count)
