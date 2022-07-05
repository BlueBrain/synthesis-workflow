"""Luigi tasks for circuit and atlas processings."""
from functools import partial

import luigi
import yaml
from atlas_analysis.planes.planes import load_planes_centerline
from atlas_analysis.planes.planes import save_planes_centerline
from luigi.parameter import OptionalPathParameter
from luigi.parameter import PathParameter
from luigi_tools.parameter import RatioParameter
from luigi_tools.task import ParamRef
from luigi_tools.task import WorkflowTask
from luigi_tools.task import copy_params
from voxcell import VoxelData

from synthesis_workflow.circuit import build_circuit
from synthesis_workflow.circuit import circuit_slicer
from synthesis_workflow.circuit import create_atlas_thickness_mask
from synthesis_workflow.circuit import create_planes
from synthesis_workflow.circuit import get_layer_tags
from synthesis_workflow.circuit import halve_atlas
from synthesis_workflow.circuit import slice_circuit
from synthesis_workflow.tasks.config import AtlasLocalTarget
from synthesis_workflow.tasks.config import CircuitConfig
from synthesis_workflow.tasks.config import CircuitLocalTarget
from synthesis_workflow.tasks.config import GetCellComposition
from synthesis_workflow.tasks.config import GetSynthesisInputs
from synthesis_workflow.tasks.config import SynthesisConfig


class CreateAtlasLayerAnnotations(WorkflowTask):
    """Create the annotation file for layers from an atlas."""

    layer_annotations_path = PathParameter(
        default="layer_annotation.nrrd",
        description=":str: Path to save layer annotations constructed from atlas.",
    )

    def requires(self):
        return GetSynthesisInputs()

    def run(self):
        """ """
        annotation, layer_mapping = get_layer_tags(
            CircuitConfig().atlas_path,
            self.input().pathlib_path / CircuitConfig().region_structure_path,
            CircuitConfig().region,
        )
        if CircuitConfig().hemisphere is not None:
            annotation.raw = halve_atlas(annotation.raw, side=CircuitConfig().hemisphere)

        annotation_path = self.output()["annotations"].path
        annotation.save_nrrd(annotation_path)

        layer_mapping_path = self.output()["layer_mapping"].path
        with open(layer_mapping_path, "w") as f:
            yaml.dump(layer_mapping, f)

    def output(self):
        """ """
        annotation_base_name = self.layer_annotations_path.stem
        layer_mapping_path = self.layer_annotations_path.with_name(
            annotation_base_name + "_layer_mapping"
        ).with_suffix(".yaml")
        return {
            "annotations": AtlasLocalTarget(self.layer_annotations_path),
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
    atlas_planes_path = PathParameter(
        default="atlas_planes", description=":str: Path to save atlas planes."
    )

    def requires(self):
        """ """
        return CreateAtlasLayerAnnotations()

    def run(self):
        """ """
        if self.plane_count < 1:
            raise Exception("Number of planes should be larger than one")

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
        """ """
        if CircuitConfig().region is not None:
            suffix = f"_{CircuitConfig().region}"
        else:
            suffix = ""
        return AtlasLocalTarget(f"{self.atlas_planes_path}{suffix}.npz")


class BuildCircuit(WorkflowTask):
    """Generate cell positions and me-types from atlas, compositions and taxonomy."""

    density_factor = RatioParameter(
        default=0.01,
        left_op=luigi.parameter.operator.lt,
        description=":float: The density of positions generated from the atlas.",
    )
    mask_path = OptionalPathParameter(
        default=None, description=":str: Path to save thickness mask (NCx only)."
    )
    seed = luigi.IntParameter(default=None, description=":int: Pseudo-random generator seed.")
    mtype_taxonomy_file = luigi.Parameter(
        default="mtype_taxonomy.tsv",
        description=":str: Filename of taxonomy file to provide to BrainBuilder",
    )

    def requires(self):
        """ """
        return {"synthesis": GetSynthesisInputs(), "composition": GetCellComposition()}

    def run(self):
        """ """
        mtype_taxonomy_path = self.input()["synthesis"].pathlib_path / self.mtype_taxonomy_file

        thickness_mask_path = None
        if self.mask_path is not None:
            thickness_mask = create_atlas_thickness_mask(CircuitConfig().atlas_path)
            thickness_mask.save_nrrd(self.mask_path)
            thickness_mask_path = self.mask_path.stem

        cells = build_circuit(
            self.input()["composition"].path,
            mtype_taxonomy_path,
            CircuitConfig().atlas_path,
            self.density_factor,
            mask=thickness_mask_path,
            seed=self.seed,
            region=CircuitConfig().region,
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

    sliced_circuit_path = PathParameter(
        default="sliced_circuit_somata.mvd3",
        description=":str: Path to save sliced circuit somata mvd3.",
    )
    n_cells = luigi.IntParameter(
        default=10, description=":int: Number of cells per mtype to consider."
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
            hemisphere=CircuitConfig().hemisphere,
        )

        cells = slice_circuit(self.input()["circuit"].path, self.output().path, _slicer)

        if len(cells.index) == 0:
            raise Exception("No cells will be synthesized, better stop here.")

    def output(self):
        """ """
        return CircuitLocalTarget(self.sliced_circuit_path)
