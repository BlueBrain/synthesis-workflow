"""Luigi tasks for circuit and atlas processings."""
from functools import partial
from pathlib import Path

import luigi
import yaml
from atlas_analysis.planes.planes import load_planes_centerline
from atlas_analysis.planes.planes import save_planes_centerline
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import LocalAtlas

from synthesis_workflow.circuit import build_circuit
from synthesis_workflow.circuit import circuit_slicer
from synthesis_workflow.circuit import create_atlas_thickness_mask
from synthesis_workflow.circuit import create_planes
from synthesis_workflow.circuit import get_centerline_bounds
from synthesis_workflow.circuit import halve_atlas
from synthesis_workflow.circuit import slice_circuit
from synthesis_workflow.tasks.config import AtlasLocalTarget
from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import CircuitLocalTarget
from synthesis_workflow.tasks.config import PathConfig
from synthesis_workflow.tasks.config import SynthesisConfig
from synthesis_workflow.tasks.luigi_tools import BoolParameter
from synthesis_workflow.tasks.luigi_tools import OptionalChoiceParameter
from synthesis_workflow.tasks.luigi_tools import ParamRef
from synthesis_workflow.tasks.luigi_tools import RatioParameter
from synthesis_workflow.tasks.luigi_tools import WorkflowTask
from synthesis_workflow.tasks.luigi_tools import copy_params
from synthesis_workflow.tasks.utils import GetSynthesisInputs
from synthesis_workflow.tools import get_layer_tags


class CreateAtlasLayerAnnotations(WorkflowTask):
    """Create the annotation file for layers from an atlas."""

    layer_annotations_path = luigi.Parameter(
        default="layer_annotation.nrrd",
        description=":str: Path to save layer annotations constructed from atlas.",
    )
    use_half = BoolParameter(
        default=False,
        description=":bool: Set to True to use half of the atlas (left or right hemisphere).",
    )
    half_axis = luigi.IntParameter(
        default=0,
        description=":int: Direction to select half of the atlas (can be 0, 1 or 2).",
    )
    half_side = luigi.IntParameter(
        default=0,
        description=":int: Side to choose to halve the atlas (0=left, 1=right).",
    )

    def run(self):
        """ """
        annotation, layer_mapping = get_layer_tags(CircuitConfig().atlas_path)

        if self.use_half:
            annotation.raw = halve_atlas(annotation.raw, axis=self.half_axis, side=self.half_side)

        annotation_path = self.output()["annotations"].path
        annotation.save_nrrd(annotation_path)

        layer_mapping_path = self.output()["layer_mapping"].path
        with open(layer_mapping_path, "w") as f:
            yaml.dump(layer_mapping, f)

    def output(self):
        """ """
        annotation_path = Path(self.layer_annotations_path)
        annotation_base_name = annotation_path.with_suffix("").name
        layer_mapping_path = annotation_path.with_name(
            annotation_base_name + "_layer_mapping"
        ).with_suffix(".yaml")
        return {
            "annotations": AtlasLocalTarget(annotation_path),
            "layer_mapping": AtlasLocalTarget(layer_mapping_path),
        }


class CreateAtlasPlanes(WorkflowTask):
    """Create plane cuts of an atlas."""

    plane_type = luigi.ChoiceParameter(
        default="centerline",
        choices=["aligned", "centerline"],
        description=(
            ":str: Type of planes creation algorithm. It can take the value 'centerline', "
            "so the center line is computed between first_bound and last_bound with internal "
            "algorithm (from atlas-analysis package), or the value 'aligned' (warning: "
            "experimental) so center line is a straight line, along the centerline_axis."
        ),
    )
    plane_count = luigi.IntParameter(
        default=10, description=":int: Number of planes to create slices of atlas."
    )
    slice_thickness = luigi.FloatParameter(
        default=100, description=":float: Thickness of slices (in micrometer)."
    )
    centerline_first_bound = luigi.ListParameter(
        default=None,
        description=(
            ":list(int): (only for plane_type == centerline) Location of first bound for "
            "centerline (in voxcell index)."
        ),
    )
    centerline_last_bound = luigi.ListParameter(
        default=None,
        description=(
            ":list(int): (only for plane_type == centerline) Location of last bound for "
            "centerline (in voxcell index)."
        ),
    )
    centerline_axis = luigi.IntParameter(
        default=0,
        description=":str: (only for plane_type = aligned) Axis along which to create planes.",
    )
    atlas_planes_path = luigi.Parameter(
        default="atlas_planes", description=":str: Path to save atlas planes."
    )
    region = luigi.Parameter(default=None, description=":str: Name of region to consider")
    hemisphere = luigi.Parameter(
        default=None, description=":str: Hemisphere to considere (left/right)"
    )

    def requires(self):
        """ """
        return CreateAtlasLayerAnnotations()

    def run(self):
        """ """
        layer_annotation = VoxelData.load_nrrd(self.input()["annotations"].path)

        atlas = LocalAtlas(CircuitConfig().atlas_path)
        if self.centerline_first_bound is None and self.centerline_last_bound is None:
            self.centerline_first_bound, self.centerline_last_bound = get_centerline_bounds(
                atlas, layer_annotation, self.region, hemisphere=self.hemisphere
            )
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
        """ """
        return AtlasLocalTarget(f"{self.atlas_planes_path}_{self.region}.npz")


@copy_params(
    mtype_taxonomy_path=ParamRef(PathConfig),
)
class BuildCircuit(WorkflowTask):
    """Generate cell positions and me-types from atlas, compositions and taxonomy.

    Attributes:
        mtype_taxonomy_path (str): Path to the taxonomy file (TSV).
    """

    cell_composition_path = luigi.Parameter(
        default="cell_composition.yaml",
        description=":str: Path to the cell composition file (YAML).",
    )
    density_factor = RatioParameter(
        default=0.01,
        left_op=luigi.parameter.operator.lt,
        description=":float: The density of positions generated from the atlas.",
    )
    mask_path = luigi.Parameter(
        default=None, description=":str: Path to save thickness mask (NCx only)."
    )
    seed = luigi.IntParameter(default=None, description=":int: Pseudo-random generator seed.")

    def requires(self):
        """ """
        return GetSynthesisInputs()

    def run(self):
        """ """
        cell_composition_path = self.input().pathlib_path / self.cell_composition_path
        mtype_taxonomy_path = self.input().pathlib_path / self.mtype_taxonomy_path

        thickness_mask_path = None
        if self.mask_path is not None:
            thickness_mask = create_atlas_thickness_mask(CircuitConfig().atlas_path)
            thickness_mask.save_nrrd(self.mask_path)
            thickness_mask_path = Path(self.mask_path).stem

        cells = build_circuit(
            cell_composition_path,
            mtype_taxonomy_path,
            CircuitConfig().atlas_path,
            self.density_factor,
            mask=thickness_mask_path,
            seed=self.seed,
        )
        cells.save(self.output().path)

    def output(self):
        """ """
        return CircuitLocalTarget(CircuitConfig().circuit_somata_path)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
)
class SliceCircuit(WorkflowTask):
    """Create a smaller circuit .mvd3 file for subsampling.

    Attributes:
        mtypes (list): List of mtypes to consider.
    """

    sliced_circuit_path = luigi.Parameter(
        default="sliced_circuit_somata.mvd3",
        description=":str: Path to save sliced circuit somata mvd3.",
    )
    n_cells = luigi.IntParameter(
        default=10, description=":int: Number of cells per mtype to consider."
    )
    hemisphere = OptionalChoiceParameter(
        default=None,
        choices=["left", "right"],
        description=":str: The hemisphere side.",
    )

    def requires(self):
        """ """
        return {
            "atlas_planes": CreateAtlasPlanes(),
            "circuit": BuildCircuit(),
        }

    def run(self):
        """ """
        planes = load_planes_centerline(self.input()["atlas_planes"].path)["planes"]

        _slicer = partial(
            circuit_slicer,
            n_cells=self.n_cells,
            mtypes=self.mtypes,
            planes=planes,
            hemisphere=self.hemisphere,
        )

        cells = slice_circuit(self.input()["circuit"].path, self.output().path, _slicer)

        if len(cells.index) == 0:
            raise Exception("No cells will be synthtesized, better stop here.")

    def output(self):
        """ """
        return CircuitLocalTarget(self.sliced_circuit_path)
