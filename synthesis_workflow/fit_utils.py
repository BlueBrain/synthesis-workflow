"""Some functions used to fit path distances with depth
"""
from typing import Tuple, Sequence

import numpy as np
import tmd
from tmd.Population.Population import Population


def _get_tmd_feature(input_population: Population, feature: str) -> np.array:
    """Returns a list of features using tmd"""
    f = [
        tmd.methods.get_persistence_diagram(n.apical[0], feature=feature)
        for n in input_population.neurons
    ]

    return np.array([np.max(p) for p in f])


def get_path_distances(input_population: Population) -> np.array:
    """Returns path distances using tmd

    Args:
        input_population: the population of neurons

    Returns: list of path distances"""
    return _get_tmd_feature(input_population, "path_distances")


def get_projections(input_population: Population) -> np.array:
    """Returns projections using tmd

    Args:
        input_population: the population of neurons

    Returns: list of projections"""
    return _get_tmd_feature(input_population, "projection")


def clean_outliers(
    x: Sequence[float], y: Sequence[float], outlier_percentage: int = 90
) -> Tuple[np.array, np.array]:
    """Returns data without outliers.

    Args:
        x: the X-axis coordinates
        y: the Y-axis coordinates
        outlier_percentage: the percentage used to find and remove outliers

    Returns: cleaned X and Y coordinates"""

    # Fit a linear function to the data
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # Detect outliers
    errs = np.array([np.abs(p(ix) - y[i]) for i, ix in enumerate(x)])
    x_clean = np.delete(
        x, [np.where(errs > np.percentile(np.sort(errs), outlier_percentage))][0]
    )
    y_clean = np.delete(
        y, [np.where(errs > np.percentile(np.sort(errs), outlier_percentage))][0]
    )

    return x_clean, y_clean


def fit_extent_to_path_distance(
    input_population: Population, outlier_percentage: int = 90
) -> Tuple[float, float]:
    """Returns two parameters (slope, intercept) for the linear fit of
    path length (X-variable) to total extents (Y-variable).
    Removes outliers up to outlier_percentage for a better fit.

    Args:
        input_population: the population of neurons
        outlier_percentage: the percentage used to find and remove outliers

    Returns: slope and intercept of the fit
    """

    # Compute path distances, projections using tmd
    x = get_path_distances(input_population)
    y = get_projections(input_population)

    # Clean data
    x_clean, y_clean = clean_outliers(x, y, outlier_percentage)

    # Get the relation between extents / path
    slope, intercept = np.polyfit(x_clean, y_clean, 1)

    return slope, intercept


def fit_path_distance_to_extent(
    input_population: Population, outlier_percentage: int = 90
) -> Tuple[float, float]:
    """Returns two parameters (slope, intercept) for the linear fit of
    path length (Y-variable) to total extents (X-variable).
    Removes outliers up to outlier_percentage for a better fit.

    Args:
        input_population: the population of neurons
        outlier_percentage: the percentage used to find and remove outliers

    Returns: slope and intercept of the fit
    """
    slope, intercept = fit_extent_to_path_distance(input_population, outlier_percentage)

    # Get the inverse to fit path to extents
    inverse_slope = 1.0 / slope
    inverse_intercept = -intercept / slope

    return inverse_slope, inverse_intercept


def get_path_distance_from_extent(
    slope: float, intercept: float, extent: float
) -> float:
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
