"""Some functions used to fit path distances with depth."""

from typing import Sequence
from typing import Tuple

import numpy as np
import tmd
from scipy.optimize import curve_fit
from tmd.Population.Population import Population


def _get_tmd_feature(
    input_population: Population, feature: str, neurite_type: str = "apical_dendrite"
) -> np.array:
    """Returns a list of features using tmd."""
    f = [
        tmd.methods.get_persistence_diagram(getattr(n, neurite_type)[0], feature=feature)
        for n in input_population.neurons
    ]

    return np.array([np.max(p) for p in f])


def get_path_distances(
    input_population: Population, neurite_type: str = "apical_dendrite"
) -> np.array:
    """Returns path distances using tmd.

    Args:
        input_population: the population of neurons

    Returns:
        list of path distances
    """
    return _get_tmd_feature(input_population, "path_distances", neurite_type)


def get_projections(
    input_population: Population, neurite_type: str = "apical_dendrite"
) -> np.array:
    """Returns projections using tmd.

    Args:
        input_population: the population of neurons

    Returns:
        list of projections
    """
    return _get_tmd_feature(input_population, "projection", neurite_type)


def fit_function(x: float, slope: float) -> float:
    """The function used to fit data."""
    return slope * x


def clean_outliers(
    x: Sequence[float], y: Sequence[float], outlier_percentage: int = 90
) -> Tuple[np.array, np.array]:
    """Returns data without outliers.

    Args:
        x: the X-axis coordinates
        y: the Y-axis coordinates
        outlier_percentage: the percentage used to find and remove outliers

    Returns:
        cleaned X and Y coordinates
    """
    # Fit a linear function passing by 0 to the data
    np.random.seed(42)  # ensure stability of fit values
    popt = curve_fit(fit_function, x, y)[0]
    p = np.poly1d([popt[0], 0])

    # Detect outliers
    errs = np.array([np.abs(p(ix) - y[i]) for i, ix in enumerate(x)])
    x_clean = np.delete(x, [np.where(errs > np.percentile(np.sort(errs), outlier_percentage))][0])
    y_clean = np.delete(y, [np.where(errs > np.percentile(np.sort(errs), outlier_percentage))][0])

    return x_clean, y_clean


def fit_path_distance_to_extent(
    input_population: Population,
    outlier_percentage: int = 90,
    neurite_type: str = "apical_dendrite",
) -> Tuple[float, float]:
    """Returns slope and intercept of a linear fit.

    Returns the two parameters (slope, intercept) for the linear fit of
    path length (Y-variable) to total extents (X-variable).
    Removes outliers up to outlier_percentage for a better fit.

    Args:
        input_population: the population of neurons
        outlier_percentage: the percentage used to find and remove outliers
        neurite_type: neurite_type to make the fit

    Returns: slope and intercept of the fit
    """
    # Compute path distances, projections using tmd
    x = get_projections(input_population, neurite_type=neurite_type)
    y = get_path_distances(input_population, neurite_type=neurite_type)

    # Clean data
    x_clean, y_clean = clean_outliers(x, y, outlier_percentage)

    # Get the relation between extents / path
    np.random.seed(42)  # ensure stability of fit values
    popt = curve_fit(fit_function, x_clean, y_clean)[0]

    # Returns the fit slope and intercept
    return popt[0], 0


def get_path_distance_from_extent(slope: float, intercept: float, extent: float) -> float:
    """Returns a path distance for an input extent according to fitted function.

    The function is given by the equation:
    Path = slope * extent + intercept

    Args:
        slope: the slope of the function
        intercept: the intercept of the function
        extent: the point where the function is evaluated

    Returns: function value evaluated at x = extent
    """
    funct = np.poly1d((slope, intercept))
    return funct(extent)
