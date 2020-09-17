"""Luigi tasks for morphology synthesis."""
from functools import partial
from pathlib import Path

import luigi
import numpy as np

from atlas_analysis.planes.planes import _create_planes
from atlas_analysis.planes.planes import create_centerline_planes
from atlas_analysis.planes.planes import save_planes_centerline
from voxcell import VoxelData

from ..circuit_slicing import generic_slicer
from ..circuit_slicing import halve_atlas
from ..circuit_slicing import slice_circuit
from ..tools import ensure_dir
from .config import circuitconfigs
from .utils import BaseTask


class CreateAtlasPlanes(BaseTask):
    """Crerate plane cuts of an atlas."""

    plane_type = luigi.Parameter(default="centerline")
    plane_count = luigi.IntParameter(default=10)
    inter_plane_count = luigi.IntParameter(default=200)

    use_half = luigi.Parameter(default=True)
    half_axis = luigi.IntParameter(default=0)
    half_side = luigi.IntParameter(default=0)

    tmp_nrrd_path = luigi.Parameter(default="layer_annotation.nrrd")

    centerline_first_bound = luigi.ListParameter(default=[126, 181, 220])
    centerline_last_bound = luigi.ListParameter(default=[407, 110, 66])

    centerline_axis = luigi.IntParameter(default=0)
    centerline_start = luigi.FloatParameter(3000)
    centerline_end = luigi.FloatParameter(10000)

    atlas_planes_path = luigi.Parameter(default="atlas_planes")

    def run(self):
        """"""
        layer_annotation_path = Path(circuitconfigs().atlas_path) / "layers.nrrd"
        layer_annotation = VoxelData.load_nrrd(layer_annotation_path)
        if self.use_half:
            layer_annotation.raw = halve_atlas(
                layer_annotation.raw, axis=self.half_axis, side=self.half_side
            )
        ensure_dir(self.tmp_nrrd_path)
        layer_annotation.save_nrrd(self.tmp_nrrd_path)

        if self.plane_type == "centerline":
            bounds = [self.centerline_first_bound, self.centerline_last_bound]
            create_centerline_planes(
                self.tmp_nrrd_path,
                self.output().path,
                bounds,
                plane_count=self.inter_plane_count,
            )

        if self.plane_type == "aligned":
            centerline = np.zeros([100, 3])
            centerline[:, self.centerline_axis] = np.linspace(
                self.centerline_start, self.centerline_end, 100
            )
            planes = _create_planes(centerline, plane_count=self.inter_plane_count)
            save_planes_centerline(self.output().path, planes, centerline)

        all_planes = np.load(self.output().path)
        selected_planes = []
        di = int(self.inter_plane_count / self.plane_count)
        for i in range(self.plane_count):
            selected_planes.append(all_planes["planes"][di * i])
            selected_planes.append(all_planes["planes"][di * i + 1])
        np.savez(
            self.output().path,
            planes=np.array(selected_planes),
            centerline=all_planes["centerline"],
            plane_format=all_planes["plane_format"],
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.atlas_planes_path + ".npz")


class SliceCircuit(BaseTask):
    """Create a smaller circuit .mvd3 file for subsampling.

    Args:
        sliced_circuit_path (str): path to save sliced circuit somata mvd3
        mtypes (list): list of mtypes to consider
        n_cells (int): number of cells per mtype to consider
        hemisphere (str): 'left' or 'right'
    """

    sliced_circuit_path = luigi.Parameter(default="sliced_circuit_somata.mvd3")
    mtypes = luigi.ListParameter(default=None)
    n_cells = luigi.IntParameter(default=10)
    hemisphere = luigi.Parameter(default=None)

    def run(self):
        """"""
        mtypes = self.mtypes
        if "all" in mtypes:  # pylint: disable=unsupported-membership-test
            mtypes = None

        if self.hemisphere is not None:
            atlas_planes = yield CreateAtlasPlanes()
            planes = np.load(atlas_planes.path)["planes"]
        else:
            planes = None

        slicer = partial(
            generic_slicer,
            n_cells=self.n_cells,
            mtypes=mtypes,
            planes=planes,
            hemisphere=self.hemisphere,
        )

        ensure_dir(self.output().path)
        cells = slice_circuit(
            circuitconfigs().circuit_somata_path, self.output().path, slicer
        )

        if len(cells.index) == 0:
            raise Exception("No cells will be synthtesized, better stop here.")

    def output(self):
        """"""
        return luigi.LocalTarget(self.sliced_circuit_path)
