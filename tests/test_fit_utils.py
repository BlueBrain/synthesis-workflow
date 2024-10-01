"""Tests for the synthesis_workflow.fit_utils module."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from tmd.Neuron import Neuron
from tmd.Population.Population import Population
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import TREE_TYPE_DICT as td

from synthesis_workflow import fit_utils


@pytest.fixture
def soma_test():
    """Fixture for a simple soma."""
    return Soma.Soma([0.0], [0.0], [0.0], [12.0])


@pytest.fixture
def apical_test():
    """Fixture for a simple apical tree."""
    return Tree.Tree(
        x=np.array(range(5, 8)),
        y=np.array(range(6, 9)),
        z=np.array(range(7, 10)),
        d=np.array([1.0, 1.0, 1.0]),
        t=np.array([4, 4, 4]),
        p=np.array([-1, 0, 1]),
    )


def test_get_features(soma_test, apical_test):
    """Test fit_utils.get_* functions."""
    neu_test = Neuron.Neuron()
    neu_test.set_soma(soma_test)
    neu_test.append_tree(apical_test, td)

    pop = Population()
    pop.append_neuron(neu_test)

    path_distances = fit_utils.get_path_distances(pop)
    projections = fit_utils.get_projections(pop)

    assert_array_almost_equal(path_distances, [3.46410162])
    assert_array_almost_equal(projections, [2])


def test_clean_data():
    """Test fit_utils.clean_outliers() function."""
    x = range(1, 10, 1)
    y = range(10, 100, 10)
    x_clean, y_clean = fit_utils.clean_outliers(x, y)
    assert_array_equal(x, x_clean)
    assert_array_equal(y, y_clean)

    # Clean spurious data
    x90 = np.concatenate([x, [5]])
    y90 = np.concatenate([y, [0]])
    x_clean, y_clean = fit_utils.clean_outliers(x90, y90)
    assert_array_equal(x, x_clean)
    assert_array_equal(y, y_clean)

    # Keep everything
    x_clean, y_clean = fit_utils.clean_outliers(x90, y90, outlier_percentage=100)
    assert_array_equal(x90, x_clean)
    assert_array_equal(y90, y_clean)


def test_fit(soma_test, apical_test):
    """Test fit_utils.fit_path_distance_to_extent() function."""
    pop = Population()
    for i in range(10):
        coeff = (i + 1) + 0.1 * i
        tree = apical_test.copy_tree()
        tree.x *= coeff
        tree.y *= coeff
        neuron = Neuron.Neuron()
        neuron.set_soma(soma_test)
        neuron.append_tree(tree, td)
        pop.append_neuron(neuron)

    slope, intercept = fit_utils.fit_path_distance_to_extent(pop)

    assert_almost_equal(slope, 1.4211942434528873)
    assert_almost_equal(intercept, 0.0)

    assert_almost_equal(fit_utils.get_path_distance_from_extent(slope, intercept, 0), intercept)
    assert_almost_equal(
        fit_utils.get_path_distance_from_extent(slope, intercept, 1), intercept + slope
    )
