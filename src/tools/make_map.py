from voxcell import CellCollection
import json
from copy import copy
from pyquaternion import Quaternion
from voxcell.voxel_data import VoxelData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.cm import get_cmap
import matplotlib


from synthesis_workflow.validation import plot_collage, get_layer_info
from bluepy.v2 import Circuit

matplotlib.use("Agg")


def get_annotations(cells, column, input_annotation):
    """From cells data, compute weighted density map."""
    annotation = copy(input_annotation)
    annotation.raw = np.array(annotation.raw, dtype=float) * 0.0

    _voxels = annotation.positions_to_indices(cells[["x", "y", "z"]].to_numpy())
    v_inds = ["_v_x", "_v_y", "_v_z"]
    tmp = pd.DataFrame(_voxels.astype(int), columns=v_inds)
    tmp.index += 1
    _values = cells[[column]].join(tmp).groupby(v_inds).mean().reset_index()
    annotation.raw[tuple(_values[v_inds].to_numpy().transpose())] = _values[
        column
    ].to_list()
    return annotation


def get_plane(annotation, axis=[0, 1, 0], angle=-0.5 * np.pi, alpha=0.4):
    """Get plane origin and rotation matrix."""
    # this is not clean, just to get a nice plane in the atlas
    bbox = annotation.bbox
    plane_origin = bbox[0] + alpha * (bbox[-1] - bbox[0])
    quaternion = Quaternion(axis=axis, angle=angle)
    rotation_matrix = quaternion.rotation_matrix
    return plane_origin, rotation_matrix


def get_sliced_annotations(
    annotation, plane_origin, rotation_matrix, n_slices=1000, thickness=0.1
):
    """From annotation file, extract values on a slice for plotting."""
    X, Y, annotations = get_layer_info(annotation, plane_origin, rotation_matrix)
    annotations[np.isnan(annotations)] = 0  # need to do that for next step

    # a single slice will have a lot of blank area, so we average over more slices
    for _ in range(n_slices):
        plane_origin += rotation_matrix.dot(np.array([0, 0, thickness]))
        _annot = get_layer_info(annotation, plane_origin, rotation_matrix)[2]
        _annot[np.isnan(_annot)] = 0
        annotations += _annot
    annotations /= n_slices

    annotations[annotations == 0] = np.nan  # set 0s back to nan for plotting
    return X, Y, annotations


def plot_density(X, Y, annotation, layer_annotation=None):
    """Make density plot with layer annotations if any."""
    plt.imshow(
        annotations.T,
        extent=[X[0, 0], X[-1, 0], Y[0, 0], Y[0, -1]],
        aspect="auto",
        origin="lower",
    )
    plt.colorbar()
    if layer_annotation is not None:
        plt.contour(
            layers.T,
            extent=[X[0, 0], X[-1, 0], Y[0, 0], Y[0, -1]],
            linewidths=0.5,
            colors="k",
        )


if __name__ == "__main__":
    cells = CellCollection.load(
        "/gpfs/bbp.cscs.ch/project/proj82/singlecell/emodel_release/nodes_emodel.h5"
    ).as_dataframe()
    layer_annotation = VoxelData.load_nrrd(
        "/gpfs/bbp.cscs.ch/project/proj82/home/arnaudon/examples_synthesis/mouse_neocortex/out/atlas/layer_annotation.nrrd"
    )
    print("data loaded")
    annotation = get_annotations(cells, "@dynamics:AIS_scaler", layer_annotation)
    annotation.save_nrrd("AIS_scaler.nrrd")
    print("density computed")

    plane_origin, rotation_matrix = get_plane(annotation)
    X, Y, annotations = get_sliced_annotations(
        annotation, plane_origin, rotation_matrix
    )
    X, Y, layers = get_layer_info(layer_annotation, plane_origin, rotation_matrix)
    print("densities sliced")

    plot_density(X, Y, annotation, layer_annotation=layers)
    plt.savefig("density_map.pdf", dpi=1000)
