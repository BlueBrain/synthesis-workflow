"""Make a atlas map from scatter data."""

from copy import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neurocollage.collage import get_annotation_info
from pyquaternion import Quaternion

matplotlib.use("Agg")


def get_annotations(cells, column, input_annotation, mode="mean"):
    """From cells data, compute weighted density map."""
    annotation = copy(input_annotation)
    annotation.raw = np.array(annotation.raw, dtype=float) * 0.0
    cells = cells.reset_index(drop=True)
    _voxels = annotation.positions_to_indices(cells[["x", "y", "z"]].to_numpy())
    v_inds = ["_v_x", "_v_y", "_v_z"]
    tmp = pd.DataFrame(_voxels.astype(int), columns=v_inds)
    if mode == "mean":
        _values = cells[[column]].join(tmp).groupby(v_inds).mean().reset_index()
    elif mode == "sum":
        _values = cells[[column]].join(tmp).groupby(v_inds).sum().reset_index()
    else:
        msg = f"Unknown mode '{mode}' (should be either 'mean' or 'sum')"
        raise ValueError(msg)
    _ids = tuple(_values[v_inds].to_numpy().transpose())
    annotation.raw[_ids] = _values[column].to_list()
    return annotation


def get_plane(annotation, axis=None, angle=-0.5 * np.pi, alpha=0.4):
    """Get plane origin and rotation matrix."""
    # this is not clean, just to get a nice plane in the atlas
    if axis is None:
        axis = [0, 1, 0]
    bbox = annotation.bbox
    plane_origin = bbox[0] + alpha * (bbox[-1] - bbox[0])
    quaternion = Quaternion(axis=axis, angle=angle)
    rotation_matrix = quaternion.rotation_matrix
    return plane_origin, rotation_matrix


def get_sliced_annotations(annotation, plane_origin, rotation_matrix, n_slices=1000, thickness=0.1):
    """From annotation file, extract values on a slice for plotting."""
    X, Y, annotations = get_annotation_info(annotation, plane_origin, rotation_matrix)
    annotations[np.isnan(annotations)] = 0  # need to do that for next step

    # a single slice will have a lot of blank area, so we average over more slices
    for _ in range(n_slices):
        plane_origin += rotation_matrix.dot(np.array([0, 0, thickness]))
        _annot = get_annotation_info(annotation, plane_origin, rotation_matrix)[2]
        _annot[np.isnan(_annot)] = 0
        annotations += _annot
    annotations /= n_slices

    annotations[annotations == 0] = np.nan  # set 0s back to nan for plotting
    return X, Y, annotations


def plot_density(X, Y, annotation, layer_annotation=None):
    """Make density plot with layer annotations if any."""
    plt.imshow(
        annotation.T,
        extent=[X[0, 0], X[-1, 0], Y[0, 0], Y[0, -1]],
        aspect="auto",
        origin="lower",
    )
    plt.colorbar()
    if layer_annotation is not None:
        plt.contour(
            layer_annotation.T,
            extent=[X[0, 0], X[-1, 0], Y[0, 0], Y[0, -1]],
            linewidths=0.5,
            colors="k",
        )
