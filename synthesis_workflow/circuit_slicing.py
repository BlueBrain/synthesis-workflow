"""Functions for slicing mvd3 circuit files to place specific cells only."""
import pandas as pd
from tqdm import tqdm
from voxcell import CellCollection

from atlas_analysis.planes.maths import Plane

LEFT = 0
RIGHT = 1


def halve_atlas(annotated_volume, axis=0, side=LEFT):
    """
    Returns the half of the annotated volume along the x-axis.

    The identifiers of the voxels located on the left half or the right half
    of the annotated volume are zeroed depending on which `side` is choosen.

    Args:
        annotated_volume: integer array of shape (W, L, D) holding the annotation
            of a brain region.
        axis: (Optional) axis along which to halve. Either 0, 1 or 2.
            Defaults to 0.
        side: (Optional) Either LEFT or RIGHT, depending on which half is requested.
            Defaults to LEFT.

    Returns:
        halves `annotated_volume` where where voxels on the opposite `side` have been
        zeroed (black).

    """
    assert axis in range(3)
    assert side in [LEFT, RIGHT]

    middle = annotated_volume.shape[axis] // 2
    slices_ = [slice(0), slice(0), slice(0)]
    for coord in range(3):
        if axis == coord:
            slices_[coord] = (
                slice(0, middle)
                if side == RIGHT
                else slice(middle, annotated_volume.shape[axis])
            )
        else:
            slices_[coord] = slice(0, annotated_volume.shape[coord])
    annotated_volume[slices_[0], slices_[1], slices_[2]] = 0
    return annotated_volume


def slice_per_mtype(cells, mtypes):
    """Select cells of given mtype."""
    return cells[cells["mtype"].isin(mtypes)]


def slice_n_cells(cells, n_cells, random_state=0):
    """Select n_cells random cells per mtypes."""
    sampled_cells = pd.DataFrame()
    for mtype in cells.mtype.unique():
        samples = cells[cells.mtype == mtype].sample(
            n=min(n_cells, len(cells[cells.mtype == mtype])), random_state=random_state
        )
        sampled_cells = sampled_cells.append(samples)
    return sampled_cells


def slice_x_slice(cells, x_slice):
    """Select cells in x_slice."""
    return cells[cells.x.between(x_slice[0], x_slice[1])]


def slice_y_slice(cells, y_slice):
    """Select cells in y_slice."""
    return cells[cells.y.between(y_slice[0], y_slice[1])]


def slice_z_slice(cells, z_slice):
    """Select cells in z_slice."""
    return cells[cells.z.between(z_slice[0], z_slice[1])]


def slice_atlas_bbox(cells, bbox):
    """Slice cells given a bbox on the atlas."""
    cells = slice_x_slice(cells, bbox[0])
    cells = slice_y_slice(cells, bbox[1])
    return slice_z_slice(cells, bbox[2])


def generic_slicer_old(cells, n_cells, mtypes=None, bbox=None):
    """Select n_cells mtype in mtypes and within bbox."""
    if mtypes is not None:
        cells = slice_per_mtype(cells, mtypes)
    if bbox is not None:
        cells = slice_atlas_bbox(cells, bbox)
    return slice_n_cells(cells, n_cells)


def generic_slicer(cells, n_cells, mtypes=None, planes=None, hemisphere=None):
    """Select n_cells mtype in mtypes."""
    if mtypes is not None:
        cells = slice_per_mtype(cells, mtypes)

    if hemisphere is not None and "hemisphere" in cells:
        cells = cells[cells.hemisphere == hemisphere]

    if planes is not None:
        # between each pair of planes, select n_cells
        return pd.concat(
            [
                slice_n_cells(
                    get_cells_between_planes(cells, plane_left, plane_right), n_cells
                )
                for plane_left, plane_right in tqdm(
                    zip(planes[:-1], planes[1:]), total=len(planes) - 1
                )
            ]
        )
    return slice_n_cells(cells, n_cells)


def is_between_planes(point, plane_left, plane_right):
    """Check if a point is between two planes in equation representation."""
    eq_left = get_plane_equation(plane_left)
    eq_right = get_plane_equation(plane_right)
    return (eq_left[:3].dot(point) > eq_left[3]) & (
        eq_right[:3].dot(point) < eq_right[3]
    )


def get_plane_equation(quaternion):
    """Get the plane equation from a quaternion representation"""
    return Plane.from_quaternion(quaternion[:3], quaternion[3:]).get_equation()


def get_cells_between_planes(cells, plane_left, plane_right):
    """Get cells gids between two planes in equation representation."""
    cells["selected"] = cells[["x", "y", "z"]].apply(
        lambda soma_position: is_between_planes(
            soma_position.to_numpy(), plane_left, plane_right
        ),
        axis=1,
    )
    return cells[cells.selected].drop("selected", axis=1)


def slice_circuit(input_mvd3, output_mvd3, slicing_function):
    """Slice an mvd3 file using a slicing function.

    Args:
        input_mvd3 (str): path to input mvd3 file
        output_mvd3 (str): path to ouput_mvd3 file
        slicing_function (function): function to slice the cells dataframe
    """
    cells = CellCollection.load_mvd3(input_mvd3)
    sliced_cells = slicing_function(cells.as_dataframe())
    sliced_cells.reset_index(inplace=True)
    sliced_cells.index += 1
    CellCollection.from_dataframe(sliced_cells).save_mvd3(output_mvd3)
    return sliced_cells
