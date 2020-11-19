"""Functions for slicing mvd3 circuit files to place specific cells only."""
import logging
import pandas as pd
from tqdm import tqdm
from voxcell import CellCollection
from voxcell.nexus.voxelbrain import LocalAtlas
import numpy as np

from atlas_analysis.planes.planes import create_planes as _create_planes
from atlas_analysis.planes.planes import create_centerline as _create_centerline
from atlas_analysis.planes.planes import _smoothing
from brainbuilder.app.cells import _place as place

L = logging.getLogger(__name__)
LEFT = 0
RIGHT = 1


def halve_atlas(annotated_volume, axis=0, side=LEFT):
    """Return the half of the annotated volume along the x-axis.

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
        Halves `annotated_volume` where where voxels on the opposite `side` have been
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


def create_atlas_thickness_mask(atlas_dir):
    """Create a mask on an atlas to to select voxels with small enough thicknesses values.

    WARNING: this function can only be used for Isocortex atlas for now.
    """
    atlas = LocalAtlas(atlas_dir)
    brain_regions = atlas.load_data("brain_regions", memcache=True)
    tolerance = 2.0 * brain_regions.voxel_dimensions[0]

    # Layer thicknesses from J. Defilipe 2017 (unpublished), see Section 5.1.1.4
    # of the release report "Neocortex Tissue Reconstruction",
    # https://github.com/BlueBrain/ncx_release_report.git
    max_thicknesses = [210.639, 190.2134, 450.6398, 242.554, 670.2, 893.62]

    isocortex_mask = atlas.get_region_mask("Isocortex", memcache=True).raw

    too_thick = np.full(isocortex_mask.shape, False, dtype=np.bool)
    for i, max_thickness in enumerate(max_thicknesses, 1):
        ph = atlas.load_data(f"[PH]{i}", memcache=True)
        with np.errstate(invalid="ignore"):
            invalid_thickness = (ph.raw[..., 1] - ph.raw[..., 0]) > (
                max_thickness + tolerance
            )
        too_thick = np.logical_or(too_thick, invalid_thickness)

        L.info(
            "Layer %s with %s percentage of bad voxels",
            i,
            np.round(100 * invalid_thickness[isocortex_mask].mean(), 3),
        )

    L.info(
        "%s percentage of bad voxels in total",
        np.round(100 * too_thick[isocortex_mask].mean(), 3),
    )

    return brain_regions.with_data(
        np.logical_and(~too_thick, isocortex_mask).astype(np.uint8)
    )


def build_circuit(
    cell_composition_path,
    mtype_taxonomy_path,
    atlas_path,
    density_factor=0.01,
    mask=None,
    seed=None,
):
    """Builds a new circuit by calling ``brainbuilder.app.cells._place``.

    Based on YAML cell composition recipe, build a circuit as MVD3 file with:
        - cell positions
        - required cell properties: 'layer', 'mtype', 'etype'
        - additional cell properties prescribed by the recipe and / or atlas
    """
    if seed is not None:
        np.random.seed(seed)
    return place(
        composition_path=cell_composition_path,
        mtype_taxonomy_path=mtype_taxonomy_path,
        atlas_url=atlas_path,
        mini_frequencies_path=None,
        atlas_cache=None,
        region=None,
        mask_dset=mask,
        density_factor=density_factor,
        soma_placement="basic",
        atlas_properties=None,
        sort_by=None,
        append_hemisphere=False,
        input_path=None,
    )


def slice_per_mtype(cells, mtypes):
    """Selects cells of given mtype."""
    return cells[cells["mtype"].isin(mtypes)]


def slice_n_cells(cells, n_cells, random_state=0):
    """Selects n_cells random cells per mtypes."""
    if n_cells <= 0:
        return cells
    sampled_cells = pd.DataFrame()
    for mtype in cells.mtype.unique():
        samples = cells[cells.mtype == mtype].sample(
            n=min(n_cells, len(cells[cells.mtype == mtype])), random_state=random_state
        )
        sampled_cells = sampled_cells.append(samples)
    return sampled_cells


def get_cells_between_planes(cells, plane_left, plane_right):
    """Gets cells gids between two planes in equation representation."""
    eq_left = plane_left.get_equation()
    eq_right = plane_right.get_equation()
    left = np.einsum("j,ij", eq_left[:3], cells[["x", "y", "z"]].values)
    right = np.einsum("j,ij", eq_right[:3], cells[["x", "y", "z"]].values)
    selected = (left > eq_left[3]) & (right < eq_right[3])
    return cells.loc[selected]


def circuit_slicer(cells, n_cells, mtypes=None, planes=None, hemisphere=None):
    """Selects n_cells mtype in mtypes."""
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
                    zip(planes[:-1:3], planes[2::3]), total=int(len(planes) / 3)
                )
            ]
        )
    return slice_n_cells(cells, n_cells)


def slice_circuit(input_mvd3, output_mvd3, slicer):
    """Slices an mvd3 file using a slicing function.

    Args:
        input_mvd3 (str): path to input mvd3 file
        output_mvd3 (str): path to ouput_mvd3 file
        slicer (function): function to slice the cells dataframe
    """
    cells = CellCollection.load_mvd3(input_mvd3)
    sliced_cells = slicer(cells.as_dataframe())
    sliced_cells.reset_index(inplace=True, drop=True)
    sliced_cells.index += 1  # this is to match CellCollection index from 1
    CellCollection.from_dataframe(sliced_cells).save_mvd3(output_mvd3)
    return sliced_cells


def create_planes(
    layer_annotation,
    plane_type="aligned",
    plane_count=10,
    slice_thickness=100,
    centerline_first_bound=None,
    centerline_last_bound=None,
    centerline_axis=0,
):
    """Create planes in an atlas.

    We create 3 * plane_count such each triplet of planes define the left, center
    and right plane of each slice.

    Args:
        layer_annotation (VoxelData): annotations with layers
        plane_type (str): type of planes creation algorithm, two choices:

            * centerline: centerline is computed between _first_bound and _last_bound with
              internal algorithm (from atlas-analysis package)
            * aligned: centerline is a straight line, along the centerline_axis

        plane_count (int): number of planes to create slices of atlas,
        slice_thickness (float): thickness of slices (in micrometer)
        centerline_first_bound (list): (for plane_type == centerline) location of first bound
            for centerline (in voxcell index)
        centerline_last_bound (list): (for plane_type == centerline) location of last bound
            for centerline (in voxcell index)
        centerline_axis (str): (for plane_type = aligned) axis along which to create planes
    """
    if plane_type == "centerline":
        centerline = _create_centerline(
            layer_annotation, [centerline_first_bound, centerline_last_bound]
        )
        centerline = _smoothing(centerline)

    elif plane_type == "aligned":
        _n_points = 10
        centerline = np.zeros([_n_points, 3])
        bbox = layer_annotation.bbox
        centerline[:, centerline_axis] = np.linspace(
            bbox[0, centerline_axis],
            bbox[1, centerline_axis],
            _n_points,
        )
    else:
        raise Exception(
            f"Please set plane_type to 'aligned' or 'centerline', not {plane_type}."
        )

    # create all planes to match slice_thickness between every two planes
    centerline_len = np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()
    total_plane_count = int(centerline_len / slice_thickness) * 2 + 1
    planes = _create_planes(centerline, plane_count=total_plane_count)

    # select plane_count planes + direct left/right neighbors
    planes_all_ids = np.arange(total_plane_count)
    id_shift = int(total_plane_count / plane_count)
    planes_select_ids = list(planes_all_ids[int(id_shift / 2) :: id_shift])
    planes_select_ids += list(planes_all_ids[int(id_shift / 2) - 1 :: id_shift])
    planes_select_ids += list(planes_all_ids[int(id_shift / 2) + 1 :: id_shift])
    return [planes[i] for i in sorted(planes_select_ids)], centerline
