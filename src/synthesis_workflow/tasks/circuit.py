"""Luigi tasks for circuit and atlas processings."""

import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path

import luigi
import pandas as pd
import yaml
from luigi.parameter import PathParameter
from luigi_tools.parameter import RatioParameter
from luigi_tools.task import ParamRef
from luigi_tools.task import WorkflowTask
from luigi_tools.task import copy_params
from neurocollage import create_planes
from neurocollage.planes import get_layer_annotation
from voxcell import VoxelData

from synthesis_workflow.circuit import build_circuit
from synthesis_workflow.circuit import circuit_slicer
from synthesis_workflow.circuit import create_boundary_mask
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
        """Required input tasks."""
        return GetSynthesisInputs()

    def run(self):
        """Actual process of the task."""
        annotation = get_layer_annotation(
            {
                "atlas": CircuitConfig().atlas_path,
                "structure": self.input().pathlib_path / CircuitConfig().region_structure_path,
            },
            CircuitConfig().region,
            CircuitConfig().hemisphere,
        )
        annotation_path = self.output()["annotations"].path
        annotation["annotation"].save_nrrd(annotation_path)

        layer_mapping_path = self.output()["layer_mapping"].path
        with open(layer_mapping_path, "w", encoding="utf-8") as f:
            yaml.dump(annotation["mapping"], f)

    def output(self):
        """Outputs of the task."""
        # pylint: disable=no-member
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
        default="aligned",
        choices=["aligned", "centerline_straight", "centerline_curved"],
        description=(
            ":str: Type of planes creation algorithm. It can take the value 'centerline', "
            "so the center line is computed between first_bound and last_bound with internal "
            "algorithm (from atlas-analysis package), or the value 'aligned' (warning: "
            "experimental) so center line is a straight line, along the centerline_axis."
        ),
    )
    plane_count = luigi.NumericalParameter(
        description=":int: Number of planes to create slices of atlas.",
        var_type=int,
        min_value=1,
        max_value=1e9,
        default=10,
    )
    slice_thickness = luigi.FloatParameter(
        default=100, description=":float: Thickness of slices (in micrometer)."
    )
    centerline_first_bound = luigi.OptionalListParameter(
        default=None,
        description=(
            ":list(int): (only for plane_type == centerline) Location of first bound for "
            "centerline (in voxcell index)."
        ),
        schema={"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
    )
    centerline_last_bound = luigi.OptionalListParameter(
        default=None,
        description=(
            ":list(int): (only for plane_type == centerline) Location of last bound for "
            "centerline (in voxcell index)."
        ),
        schema={"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
    )
    centerline_axis = luigi.IntParameter(
        default=0,
        description=":str: (only for plane_type = aligned) Axis along which to create planes.",
    )
    atlas_planes_path = PathParameter(
        default="atlas_planes", description=":str: Path to save atlas planes."
    )

    def requires(self):
        """Required input tasks."""
        return CreateAtlasLayerAnnotations()

    def run(self):
        """Actual process of the task."""
        if self.plane_count < 1:
            raise ValueError("Number of planes should be larger than one")

        layer_annotation = VoxelData.load_nrrd(self.input()["annotations"].path)
        layer_mappings = yaml.safe_load(self.input()["layer_mapping"].open())

        planes, centerline = create_planes(
            {"annotation": layer_annotation, "mapping": layer_mappings},
            self.plane_type,
            self.plane_count,
            self.slice_thickness,
            self.centerline_first_bound,
            self.centerline_last_bound,
            self.centerline_axis,
        )

        with open(self.output().path, "wb") as f_planes:
            pickle.dump({"planes": planes, "centerline": centerline}, f_planes)

    def output(self):
        """Outputs of the task."""
        if CircuitConfig().region is not None:
            suffix = f"_{CircuitConfig().region}"
        else:
            suffix = ""
        return AtlasLocalTarget(f"{self.atlas_planes_path}{suffix}.npz")


class CreateBoundaryMask(WorkflowTask):
    """Create mask at the boundary of the region to prevent placing somata too close to boundary."""

    boundary_thickness = luigi.IntParameter(
        default=0,
        description=":int: Thickness to create a mask to prevent placing cells near boundary"
        ", in units of voxel size.",
    )
    mask_path = luigi.Parameter(default="boundary_mask.nrrd")

    def requires(self):
        """Required input tasks."""
        return GetSynthesisInputs()

    def run(self):
        """Actual process of the task."""
        thickness_mask = create_boundary_mask(
            {
                "atlas": CircuitConfig().atlas_path,
                "structure": self.input().pathlib_path / CircuitConfig().region_structure_path,
            },
            CircuitConfig().region,
            self.boundary_thickness,
        )
        thickness_mask.save_nrrd(self.output().path)

    def output(self):
        """Outputs of the task."""
        return CircuitLocalTarget(self.mask_path)


class BuildCircuit(WorkflowTask):
    """Generate cell positions and me-types from atlas, compositions and taxonomy."""

    density_factor = RatioParameter(
        default=0.5,
        left_op=luigi.parameter.operator.lt,
        description=":float: The density of positions generated from the atlas.",
    )
    seed = luigi.OptionalIntParameter(default=42, description=":int: Pseudo-random generator seed.")
    mtype_taxonomy_file = luigi.Parameter(
        default="mtype_taxonomy.tsv",
        description=":str: Filename of taxonomy file to provide to BrainBuilder",
    )

    def requires(self):
        """Required input tasks."""
        tasks = {"synthesis": GetSynthesisInputs(), "composition": GetCellComposition()}
        if CreateBoundaryMask().boundary_thickness > 0:
            tasks["boundary"] = CreateBoundaryMask()

        return tasks

    def run(self):
        """Actual process of the task."""
        # pylint: disable=no-member
        mtype_taxonomy_path = self.input()["synthesis"].pathlib_path / self.mtype_taxonomy_file

        # create uniform densities if they don't exist
        mtypes = SynthesisConfig().mtypes
        if not mtypes:
            mtypes = pd.read_csv(mtype_taxonomy_path, sep="\t")["mtype"].to_list()
        composition = yaml.safe_load(self.input()["composition"].open())["neurons"]

        for mtype in mtypes:
            nrrd_path = Path("")
            for data in composition:
                if data["traits"]["mtype"] == mtype:
                    name = "".join(list(data["density"])[1:-1])
                    nrrd_path = Path(CircuitConfig().atlas_path) / f"{name}.nrrd"
            if not nrrd_path.exists():
                annotation = get_layer_annotation(
                    {
                        "atlas": CircuitConfig().atlas_path,
                        "structure": self.input()["synthesis"].pathlib_path
                        / CircuitConfig().region_structure_path,
                    },
                    CircuitConfig().region,
                    CircuitConfig().hemisphere,
                )
                layer = mtype[1]
                keys = [k + 1 for k, d in annotation["mapping"].items() if d.endswith(layer)]
                density_annotation = deepcopy(annotation["annotation"])
                density_annotation.raw[annotation["annotation"].raw == keys[0]] = 100000
                density_annotation.raw[annotation["annotation"].raw != keys[0]] = 0
                density_annotation.save_nrrd(str(nrrd_path))

        cells = build_circuit(
            self.input()["composition"].path,
            mtype_taxonomy_path,
            CircuitConfig().atlas_path,
            self.density_factor,
            mask=(
                self.input()["boundary"].path
                if CreateBoundaryMask().boundary_thickness > 0
                else None
            ),
            seed=self.seed,
            region=CircuitConfig().region,
        )
        # new version of brainbuilder only assigns subregion, not sure why, to investigate
        if "region" not in cells.properties:
            cells.properties["region"] = cells.properties["subregion"]
        cells.save(self.output().path)

    def output(self):
        """Outputs of the task."""
        return CircuitLocalTarget(CircuitConfig().circuit_somata_path)


@copy_params(
    mtypes=ParamRef(SynthesisConfig),
)
class SliceCircuit(WorkflowTask):
    """Create a smaller circuit file for subsampling.

    Attributes:
        mtypes (list): List of mtypes to consider.
    """

    sliced_circuit_path = PathParameter(
        default="sliced_circuit_somata.h5",
        description=":str: Path to save sliced circuit.",
    )
    n_cells = luigi.IntParameter(
        default=10, description=":int: Number of cells per mtype to consider."
    )

    def requires(self):
        """Required input tasks."""
        return {
            "atlas_planes": CreateAtlasPlanes(),
            "circuit": BuildCircuit(),
        }

    def run(self):
        """Actual process of the task."""
        with open(self.input()["atlas_planes"].path, "rb") as f_planes:
            planes = pickle.load(f_planes)["planes"]

        _slicer = partial(
            circuit_slicer,
            n_cells=self.n_cells,
            mtypes=self.mtypes,
            planes=planes,
            hemisphere=CircuitConfig().hemisphere,
        )

        cells = slice_circuit(self.input()["circuit"].path, self.output().path, _slicer)

        if len(cells.index) == 0:
            raise ValueError("No cells will be synthesized, better stop here.")

    def output(self):
        """Outputs of the task."""
        return CircuitLocalTarget(self.sliced_circuit_path)
