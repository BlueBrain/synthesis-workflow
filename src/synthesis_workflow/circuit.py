"""Functions for slicing mvd3 circuit files to place specific cells only."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from brainbuilder.app.cells import _place as place
from neurocollage.mesh_helper import MeshHelper
from neurocollage.planes import get_cells_between_planes
from neurocollage.planes import slice_n_cells
from neurocollage.planes import slice_per_mtype
from region_grower.atlas_helper import AtlasHelper
from trimesh.voxel import VoxelGrid
from voxcell import CellCollection
from voxcell.nexus.voxelbrain import LocalAtlas

L = logging.getLogger(__name__)


def create_boundary_mask(atlas_dir, region, boundary_thickness=10):
    """Create a mask on an atlas to to select voxels away from boundary with some voxel distance."""
    atlas = LocalAtlas(atlas_dir["atlas"])
    mesh = MeshHelper(atlas_dir, region)

    region_mask = atlas.get_region_mask(region, memcache=True)

    boundary_mesh = mesh.get_boundary_mesh()
    data = np.zeros_like(region_mask.raw, dtype=int)
    data_vg = VoxelGrid(data)
    # pylint: disable=no-member
    data[tuple(data_vg.points_to_indices(boundary_mesh.triangles_center).T)] = 1
    for _ in range(boundary_thickness):
        data[:, :, :-1] += data[:, :, 1:]
        data[:, :, 1:] += data[:, :, :-1]
        data[:, :-1] += data[:, 1:]
        data[:, 1:] += data[:, :-1]
        data[:-1] += data[1:]
        data[1:] += data[:-1]

    mask = 1 - np.clip(data, 0, 1)
    region_mask = atlas.get_region_mask(region, memcache=True)
    mask[~region_mask.raw] = 0
    return region_mask.with_data(mask.astype(np.uint8))


def get_regions_from_composition(cell_composition_path):
    """Get list of region regex from cell_composition."""
    with open(cell_composition_path, "r", encoding="utf-8") as comp_p:
        cell_composition = yaml.safe_load(comp_p)

    region_regex = "@"
    for reg in set(entry["region"] for entry in cell_composition["neurons"]):
        if reg[-1] != "|":
            reg += "|"
        if reg[0] == "@":
            region_regex += reg[1:]
        else:
            region_regex += reg
    return region_regex[:-1]


def build_circuit(
    cell_composition_path,
    mtype_taxonomy_path,
    atlas_path,
    density_factor=0.01,
    mask=None,
    seed=None,
    region=None,
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
        region=region or get_regions_from_composition(cell_composition_path),
        mask_dset=Path(mask).resolve().with_suffix("") if mask else None,
        density_factor=density_factor,
        soma_placement="basic",
        atlas_properties=None,
        sort_by=None,
        append_hemisphere=False,
        input_path=None,
    )


def circuit_slicer(cells, n_cells, mtypes=None, planes=None, hemisphere=None):
    """Selects n_cells mtype in mtypes."""
    if mtypes is not None:
        cells = slice_per_mtype(cells, mtypes)

    # TODO: rough way to split hemisphere, maybe there is a better way, to investigate if needed
    if hemisphere is not None:
        if hemisphere == "left":
            cells = cells[cells.z < cells.z.mean()]
        if hemisphere == "right":
            cells = cells[cells.z >= cells.z.mean()]

    if planes is not None:
        # between each pair of planes, select n_cells
        return pd.concat(
            [
                slice_n_cells(
                    get_cells_between_planes(cells, plane["left"], plane["right"]), n_cells
                )
                for plane in planes
            ]
        )
    return slice_n_cells(cells, n_cells)


def slice_circuit(input_circuit, output_circuit, slicer):
    """Slices a circuit file using a slicing function.

    Args:
        input_circuit (str): path to input file
        output_circuit (str): path to output file
        slicer (function): function to slice the cells dataframe
    """
    cells = CellCollection.load(input_circuit)
    sliced_cells = slicer(cells.as_dataframe())
    sliced_cells.reset_index(inplace=True, drop=True)
    sliced_cells.index += 1  # this is to match CellCollection index from 1
    CellCollection.from_dataframe(sliced_cells).save(output_circuit)
    return sliced_cells


def _get_principal_direction(points):
    """Return the principal direction of a point cloud.

    It is the eigen vector of the covariance matrix with the highest eigen value.
    Taken from neuror.unravel.
    """
    X = np.copy(np.asarray(points))
    X -= np.mean(X, axis=0)
    C = np.dot(X.T, X)
    w, v = np.linalg.eig(C)
    return v[:, w.argmax()]


def get_centerline_bounds(layer):
    """Find centerline bounds using PCA of the voxell position of a given layer in the region."""
    _ls = np.unique(layer.raw[layer.raw > 0])
    central_layer = _ls[int(len(_ls) / 2)]

    # we select voxels which are on the boundary of the region, to prevent picking them in y dir
    layer_raw = np.array(layer.raw, dtype=float)
    layer_raw[layer_raw == 0] = -10000  # large number to be safe
    boundary_mask = sum(abs(g) for g in np.gradient(layer_raw)) > 1000
    ids = np.column_stack(np.where((layer.raw == central_layer) & boundary_mask))
    points = layer.indices_to_positions(ids)
    _align = points.dot(_get_principal_direction(points))
    return ids[_align.argmin()], ids[_align.argmax()]


def get_local_bbox(annotation):
    """Compute bbox where annotation file is strictly positive."""
    ids = np.where(annotation.raw > 0)
    dim = annotation.voxel_dimensions
    return annotation.offset + np.array(
        [np.min(ids, axis=1) * dim, (np.max(ids, axis=1) + 1) * dim]
    )


def get_layer_tags(atlas_dir, region_structure_path, region=None):
    """Create a VoxelData with layer tags."""
    atlas_helper = AtlasHelper(LocalAtlas(atlas_dir), region_structure_path=region_structure_path)

    brain_regions = atlas_helper.brain_regions
    layers_data = np.zeros_like(brain_regions.raw, dtype="uint8")

    region_mask = None
    if region is not None:
        region_mask = atlas_helper.atlas.get_region_mask(region).raw

    layer_mapping = {}
    for layer_id, layer in enumerate(atlas_helper.layers):
        layer_mapping[layer_id] = atlas_helper.region_structure["names"].get(layer, str(layer))
        region_query = atlas_helper.region_structure["region_queries"].get(layer, None)
        mask = atlas_helper.atlas.get_region_mask(region_query).raw
        if region_mask is not None:
            mask *= region_mask
        layers_data[mask] = layer_id + 1
        if not len(layers_data[mask]):
            L.warning("No voxel found for layer %s.", layer)
    brain_regions.raw = layers_data
    return brain_regions, layer_mapping
