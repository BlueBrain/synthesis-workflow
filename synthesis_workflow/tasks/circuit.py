"""Luigi tasks for morphology synthesis."""
from functools import partial
from pathlib import Path
import yaml

import luigi
import numpy as np

from voxcell import VoxelData
from voxcell.nexus.voxelbrain import LocalAtlas
from atlas_analysis.planes.planes import load_planes_centerline
from atlas_analysis.planes.planes import save_planes_centerline

from ..circuit import build_circuit
from ..circuit import circuit_slicer
from ..circuit import create_planes
from ..circuit import halve_atlas
from ..circuit import slice_circuit
from ..tools import ensure_dir
from .config import CircuitConfig
from .config import PathConfig
from .config import SynthesisConfig
from .luigi_tools import BoolParameter
from .luigi_tools import copy_params
from .luigi_tools import OutputLocalTarget
from .luigi_tools import ParamLink
from .luigi_tools import WorkflowTask


class CreateAtlasLayerAnnotations(WorkflowTask):
    """Create the annotation file for layers from an atlas.

    Args:
        layer_annotations_path (str): path to save layer annotations constructed from atlas
        use_half (bool): set to True to use half of the atlas (left or right hemisphere)
        half_axis (int): direction to select half of the atlas (can be 0, 1 or 2)
        half_side (int): side to choose to halve the atlas (0=left, 1=right)
    """

    layer_annotations_path = luigi.Parameter(default="layer_annotation.nrrd")
    use_half = BoolParameter(default=False)
    half_axis = luigi.IntParameter(default=0)
    half_side = luigi.IntParameter(default=0)

    def run(self):
        """"""
        atlas = LocalAtlas(CircuitConfig().atlas_path)
        ids, names = atlas.get_layers()  # pylint: disable=no-member
        annotation = VoxelData.load_nrrd(
            Path(CircuitConfig().atlas_path) / "brain_regions.nrrd"
        )
        layers = np.zeros_like(annotation.raw, dtype="uint8")
        layer_mapping = {}
        for layer_id, (ids_set, layer) in enumerate(zip(ids, names)):
            layer_mapping[layer_id] = layer
            layers[np.isin(annotation.raw, list(ids_set))] = layer_id + 1
        annotation.raw = layers

        if self.use_half:
            annotation.raw = halve_atlas(
                annotation.raw, axis=self.half_axis, side=self.half_side
            )

        annotation_path = self.output()["annotations"].path
        ensure_dir(annotation_path)
        annotation.save_nrrd(annotation_path)

        layer_mapping_path = self.output()["layer_mapping"].path
        ensure_dir(layer_mapping_path)
        yaml.dump(
            layer_mapping,
            open(
                layer_mapping_path,
                "w",
            ),
        )

    def output(self):
        """"""
        annotation_path = Path(self.layer_annotations_path)
        annotation_base_name = annotation_path.with_suffix("").name
        layer_mapping_path = annotation_path.with_name(
            annotation_base_name + "_layer_mapping"
        ).with_suffix(".yaml")
        return {
            "annotations": OutputLocalTarget(annotation_path),
            "layer_mapping": OutputLocalTarget(layer_mapping_path),
        }


class CreateAtlasPlanes(WorkflowTask):
    """Create plane cuts of an atlas.

    Args:
        plane_type (str): type of planes creation algorithm, two choices:
            - centerline: centerline is computed between _first_bound and _last_bound with
                internal algorithm (from atlas-analysis package), (warning: experimental)
            - aligned: centerline is a straight line, along the centerline_axis
        plane_count (int): number of planes to create slices of atlas,
        slice_thickness (float): thickness of slices (in micrometer)
        centerline_first_bound (list): (for plane_type == centerline) location of first bound
            for centerline (in voxcell index)
        centerline_last_bound (list): (for plane_type == centerline) location of last bound
            for centerline (in voxcell index)
        centerline_axis (str): (for plane_type = aligned) axis along which to create planes
        atlas_planes_path (str): path to save atlas planes
    """

    plane_type = luigi.ChoiceParameter(
        default="centerline", choices=["aligned", "centerline"]
    )
    plane_count = luigi.IntParameter(default=10)
    slice_thickness = luigi.FloatParameter(default=100)

    centerline_first_bound = luigi.ListParameter(default=[126, 181, 220])
    centerline_last_bound = luigi.ListParameter(default=[407, 110, 66])
    centerline_axis = luigi.IntParameter(default=0)

    atlas_planes_path = luigi.Parameter(default="atlas_planes")

    def requires(self):
        """"""
        return CreateAtlasLayerAnnotations()

    def run(self):
        """"""
        layer_annotation = VoxelData.load_nrrd(self.input()["annotations"].path)
        planes, centerline = create_planes(
            layer_annotation,
            self.plane_type,
            self.plane_count,
            self.slice_thickness,
            self.centerline_first_bound,
            self.centerline_last_bound,
            self.centerline_axis,
        )

        save_planes_centerline(self.output().path, planes, centerline)

    def output(self):
        """"""
        return OutputLocalTarget(self.atlas_planes_path + ".npz")


@copy_params(
    mtype_taxonomy_path=ParamLink(PathConfig),
)
class BuildCircuit(WorkflowTask):
    """Generate cell positions and me-types from atlas, compositions and taxonomy.

    Args:
        cell_composition_path (str): path to the cell composition file (YAML)
        mtype_taxonomy_path (str): path to the taxonomy file (TSV)
        density_factor (float): density factor
        seed (int): pseudo-random generator seed
    """

    cell_composition_path = luigi.Parameter(
        description="path to the cell composition file (YAML)"
    )
    density_factor = luigi.NumericalParameter(
        default=0.01,
        var_type=float,
        min_value=0,
        max_value=1,
        left_op=luigi.parameter.operator.lt,
        right_op=luigi.parameter.operator.le,
        description="The density of positions generated from the atlas",
    )
    seed = luigi.IntParameter(default=None, description="pseudo-random generator seed")

    def run(self):
        """"""
        cells = build_circuit(
            self.cell_composition_path,
            self.mtype_taxonomy_path,
            CircuitConfig().atlas_path,
            self.density_factor,
            self.seed,
        )
        ensure_dir(self.output().path)
        cells.save(self.output().path)

    def output(self):
        """"""
        return OutputLocalTarget(CircuitConfig().circuit_somata_path)


@copy_params(
    mtypes=ParamLink(SynthesisConfig),
)
class SliceCircuit(WorkflowTask):
    """Create a smaller circuit .mvd3 file for subsampling.

    Args:
        sliced_circuit_path (str): path to save sliced circuit somata mvd3
        mtypes (list): list of mtypes to consider
        n_cells (int): number of cells per mtype to consider
        hemisphere (str): 'left' or 'right'
    """

    sliced_circuit_path = luigi.Parameter(default="sliced_circuit_somata.mvd3")
    n_cells = luigi.IntParameter(default=10)
    hemisphere = luigi.Parameter(default=None)

    def requires(self):
        """"""
        return {
            "atlas_planes": CreateAtlasPlanes(),
            "circuit": BuildCircuit(),
        }

    def run(self):
        """"""
        mtypes = self.mtypes
        if (
            # pylint: disable=unsupported-membership-test
            mtypes is None
            or "all" in mtypes
        ):
            mtypes = None

        planes = load_planes_centerline(self.input()["atlas_planes"].path)["planes"]

        _slicer = partial(
            circuit_slicer,
            n_cells=self.n_cells,
            mtypes=mtypes,
            planes=planes,
            hemisphere=self.hemisphere,
        )

        ensure_dir(self.output().path)
        cells = slice_circuit(self.input()["circuit"].path, self.output().path, _slicer)

        if len(cells.index) == 0:
            raise Exception("No cells will be synthtesized, better stop here.")

    def output(self):
        """"""
        return OutputLocalTarget(self.sliced_circuit_path)